import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gymtorax.action_handler as ah
import gymtorax.observation_handler as oh
import gymtorax.rewards as rw
from gymtorax import IterHybridEnv

# Set up logger for this module
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 25})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

class PIDAgent:
    def __init__(self, action_space, kp, ki, kd):
        self.action_space = action_space
        self.time = 0

        # PID state variables
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.dt = 1.0  # Time step in seconds (from config: fixed_dt = 1)

        # Anti-windup for ramp rate limiting
        self.anti_windup_enabled = True  # Enable/disable ramp rate anti-windup

        # Control limits
        self.ip_controlled = 0  # Current controlled power

        # Physical power constraints
        self.ip_min = 0.001e6  # Minimum Ip current: 0 MA
        self.ip_max = 15e6  # Maximum Ip power: 15 MA

        # Tracking variables for plotting
        self.j_target_history = []
        self.j_actual_history = []
        self.time_history = []
        self.error_history = []
        self.action_history = []

    def act(self, observation, j_target) -> dict:
        j_center = observation["profiles"]["j_total"][0]

        # Store values for tracking/plotting
        self.j_target_history.append(j_target)
        self.j_actual_history.append(j_center)
        self.time_history.append(self.time)

        if self.time >= 100:
            # keep the same self.ip_controlled after 100s
            pass
        else:
            # Calculate PID error (desired - actual)
            error = j_target - j_center
            self.error_history.append(error)

            # Derivative term (rate of change of error)
            if self.time > 0:
                error_derivative = (error - self.previous_error) / self.dt
            else:
                error_derivative = 0.0

            # Calculate PID control components (before updating integral)
            p_term = self.kp * error
            i_term = self.ki * self.error_integral
            d_term = self.kd * error_derivative

            # PID control output
            pid_output = p_term + i_term + d_term

            # Calculate desired Ip current (baseline + PID correction)
            ip_baseline = 3.0e6
            ip_desired = ip_baseline + pid_output

            # Debug prints
            # if self.time % 5 == 0:  # Print every 10 time steps
            #     print(
            #         f"[PID DEBUG] t={self.time:3d}  j_obj={j_target:.3e}  j={j_center:.3e}  error={error:.3e}"
            #     )
            #     print(
            #         f"            P={p_term:.3e}  I={i_term:.3e}  D={d_term:.3e}  PID_out={pid_output:.3e}  I_int={self.error_integral:.3e}"
            #     )
            #     print(
            #         f"            Ip_ctrl={self.ip_controlled:.3e}  Ip_des={ip_desired:.3e}\n"
            #     )

            # Then apply physical power limits
            ip_final = np.clip(ip_desired, self.ip_min, self.ip_max)

            # Apply ramp rate limiting (0.2 MA/s = 0.2e6 A/s)
            max_ramp_rate = 0.2e6  # A/s
            max_change = max_ramp_rate * self.dt  # Maximum change per time step

            is_ramp_limited = False
            if self.time > 0:  # Only apply ramp rate limiting after first step
                ip_change = ip_final - self.ip_controlled
                if abs(ip_change) > max_change:
                    is_ramp_limited = True
                    # Limit the change to the maximum ramp rate
                    ip_final = self.ip_controlled + np.sign(ip_change) * max_change

            # Check what type of limiting is occurring
            is_power_limited = ip_final != ip_desired

            # Anti-windup: only update integral if not limited, or if error would help
            if self.anti_windup_enabled and (is_power_limited or is_ramp_limited):
                # Determine what type of limit we're hitting
                hitting_upper_power = ip_final > self.ip_max
                hitting_lower_power = ip_final < self.ip_min

                # Only integrate if error would help reduce the limiting
                any_limiting = hitting_upper_power or hitting_lower_power
                if not any_limiting:
                    self.error_integral += error * self.dt
            else:
                # Standard integral update (no anti-windup or not limited)
                self.error_integral += error * self.dt

            # Update the controlled value
            self.ip_controlled = ip_final

            # Store current error for next derivative calculation
            self.previous_error = error

        # Create action dictionary
        action = {"Ip": [self.ip_controlled]}
        self.time += 1
        self.action_history.append(self.ip_controlled)

        return action

    def plot_j_evolution(self, filename):
        """Plot the evolution of j_target and j_actual over time.

        Args:
            filename (str, optional): If provided, save the plot to this file
        """
        fig, ax = plt.subplots()

        j_target_ma = np.array(self.j_target_history) / 1e6
        j_actual_ma = np.array(self.j_actual_history) / 1e6

        # Plot j_target and j_actual
        ax.plot(
            self.time_history,
            j_actual_ma,
            "b-",
            label=r"$j_{actual}$",
            linewidth=2,
        )

        ax.plot(
            self.time_history[:100],
            j_target_ma[:100],
            "r--",
            label=r"$j_{target}$",
            linewidth=1.5,
        )

        # Add ±10% relative error lines
        j_target_plus_10 = j_target_ma * 1.1
        j_target_minus_10 = j_target_ma * 0.9

        ax.plot(
            self.time_history[:100],
            j_target_plus_10[:100],
            "r:",
            label=r"$\pm$ $10\,\%$",
            linewidth=1,
        )
        ax.plot(
            self.time_history[:100],
            j_target_minus_10[:100],
            "r:",
            linewidth=1,
        )

        # Add vertical line at t=100 with LH transition text
        ax.axvline(x=100, linestyle="dashed", color="black", linewidth=1)
        ax.text(
            105,
            (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05 + ax.get_ylim()[0],
            "LH transition",
            fontsize=20,
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current density (MA/m²)")
        ax.legend(prop={"size": 20})

        fig.savefig(f"{filename}_error.pdf", bbox_inches="tight")

        fig, ax = plt.subplots()
        ax.plot(
            self.time_history,
            np.array(self.action_history) / 1e6,
            linewidth=2,
            color="blue",
            label=r"$I_p$",
        )

        # Add vertical line at t=100 with LH transition text for Ip plot too
        ax.axvline(x=100, linestyle="dashed", color="black", linewidth=1)
        ax.text(
            105,
            (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05 + ax.get_ylim()[0],
            "LH transition",
            fontsize=20,
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total current (MA)")
        ax.legend(prop={"size": 20})

        fig.savefig(f"{filename}_ip.pdf", bbox_inches="tight")


class IterHybridEnvPid(IterHybridEnv):  # noqa: D101
    def __init__(self, render_mode, **kwargs):  # noqa: D107
        super().__init__(render_mode=render_mode, **kwargs)

    @property
    def _define_actions(self):  # noqa: D102
        actions = [ah.IpAction()]

        return actions

    @property
    def _define_observation(self):  # noqa: D102
        return oh.AllObservation(custom_bounds_file="examples/iter_hybrid.json")

    def _define_reward(self, state, next_state, action):
        """Compute the reward. The higher the reward, the more performance and stability of the plasma.

        The reward is a weighted sum of several factors:
        - Fusion gain fusion_gain: we want to maximize it.
        - Beta_N: we want to be as close as possible to the Troyon limit, but not exceed it too much.
        - H-mode confinement quality factor h98: great if > 1
        - q_min: we want to avoid it to be below 1.
        - q_edge: we want to avoid it to be below 3.
        - Magnetic shear at rational surfaces: we want to avoid low shear at rational surfaces.

        Args:
            state (dict[str, Any]): state at time t
            next_state (dict[str, Any]): state at time t+1
            action (NDArray[np.floating]): applied action at time t
            gamma (float): discounted factor (0 < gamma <= 1)
            n (int): number of steps since the beginning of the episode

        Returns:
            float: reward associated to the transition (state, action, next_state)
        """
        fusion_gain = rw.get_fusion_gain(next_state) / 10  # Normalize to [0, 1]
        beta_N = rw.get_beta_N(next_state)
        h98 = rw.get_h98(next_state)
        q_profile = rw.get_q_profile(next_state)
        q_min = rw.get_q_min(next_state)
        q_95 = rw.get_q95(next_state)
        s_profile = rw.get_s_profile(next_state)
        j_center = rw.get_j_profile(next_state)[0]

        # Customize weights and sigma as needed
        weight_list = [2, 1, 2, 1, 1, 1]
        sigma = 0.5

        def gaussian_beta():
            """Compute the Gaussian weight for the beta_N value.

            Beta_N is a measure of performance and instabilities in plasma physics.
            We have to find a trade-off between stability and performance.
            The Troyon limit is an empirical limit which can be used as a trade-off.
            However, it can be exceeded in certain scenarios.
            For this reason, we need a correlation to allow a slight excess.
            That's why we use a Gaussian function.
            """
            Ip = action["Ip"][0] / 1e6  # We know that Ip is in action
            beta_troyon = (
                0.028
                * Ip
                / (
                    next_state["scalars"]["a_minor"][0]
                    * next_state["scalars"]["B_0"][0]
                )
            )
            return np.exp(-0.5 * ((beta_N - beta_troyon) / sigma) ** 2)

        def q_min_function():
            if q_min <= 1:
                return 0
            elif q_min > 1:
                return 1

        def q_edge_function():
            if q_95 <= 3:
                return 0
            else:
                return 1

        def s_function():
            q_risk = [1, 4 / 3, 3 / 2, 5 / 3, 2, 5 / 2, 3]
            weight = [-1, -1, -1, -1, -1, -1, -1]
            s0 = 0.1  # Value to avoid a division by zero
            sum_ = 0
            q_max = max(q_profile)

            for q_val, w_val in zip(q_risk, weight):
                if not (q_min <= q_val <= q_max):
                    continue

                for i in range(len(q_profile) - 1):
                    q1, q2 = q_profile[i], q_profile[i + 1]
                    s1, s2 = s_profile[i], s_profile[i + 1]

                    if (q1 <= q_val <= q2) or (q2 <= q_val <= q1):
                        # interpolate s from the estimated position q
                        s = np.interp(q_val, [q1, q2], [s1, s2])
                        sum_ += w_val * 1 / (abs(s) + s0)
            return sum_

        def is_H_mode():
            if (
                next_state["profiles"]["T_e"][0] > 10
                and next_state["profiles"]["T_i"][0] > 10
            ):
                return True
            else:
                return False

        # Calculate individual reward components
        r_fusion_gain = weight_list[0] * fusion_gain / 50 if is_H_mode() else 0
        r_beta = weight_list[1] * gaussian_beta()
        r_h98 = weight_list[2] * 1 / 50 if (is_H_mode() and h98 >= 1) else 0
        r_q_min = weight_list[3] * q_min_function() / 150
        r_q_edge = weight_list[4] * q_edge_function() / 150
        r_mag_shear = weight_list[5] * s_function()

        total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_edge

        # Store reward breakdown for logging (attach to environment if it has reward_breakdown attribute)
        if hasattr(self, "reward_breakdown"):
            if not hasattr(self, "_reward_components"):
                self._reward_components = {
                    "fusion_gain": [],
                    # "beta_N": [],
                    "h98": [],
                    "q_min": [],
                    "q_edge": [],
                    # "s_function": [],
                }

            self._reward_components["fusion_gain"].append(r_fusion_gain)
            # self._reward_components["beta_N"].append(r_beta)
            self._reward_components["h98"].append(r_h98)
            self._reward_components["q_min"].append(r_q_min)
            self._reward_components["q_edge"].append(r_q_edge)
            # self._reward_components["s_function"].append(r_mag_shear)

        return total_reward

    def _j_objectif(self):
        """Compute the objective function for the current density.

        Returns:
            float: The objective function value.
        """
        return (
            0.2e6 + 0.4e6 + 1.4e6 * self.current_time / 100
            if self.current_time <= 100
            else 2e6
        )


def simulate(env: IterHybridEnvPid, k, save_plot_as=None):
    """Simulate the environment with given PID parameters and return a cost.

    Args:
        env: The environment to simulate
        k: PID parameters [kp, ki]
        save_plot_as: Filename to save the plot (if plotting is enabled)

    Returns:
        float: Negative cumulative reward (cost to minimize)
    """
    kp, ki = k

    agent = PIDAgent(env.action_space, kp=kp, ki=ki, kd=0.0)

    # Enable reward breakdown tracking
    env.reward_breakdown = True

    # Clear any previous reward components to avoid accumulation between optimization iterations
    if hasattr(env, "_reward_components"):
        delattr(env, "_reward_components")

    observation, _ = env.reset()
    terminated = False
    cumulative_reward = 0.0
    n = 0
    gamma = 1  # Discount factor for future rewards

    while not terminated:
        action = agent.act(observation, env._j_objectif())
        observation, reward, terminated, _, _ = env.step(action)

        cumulative_reward += reward * (gamma**n)
        n += 1

    # Calculate and log reward component shares
    if hasattr(env, "_reward_components"):
        pos_component_totals = {
            name: sum(values)
            for name, values in env._reward_components.items()
            if name != "j_error"
        }
        pos_total_sum = sum(pos_component_totals.values())

        # print(f"Simulation completed with kp={kp:.4e}, ki={ki:.4e}")
        print(
            f"Total cumulative reward: {cumulative_reward:.4f} with kp={kp:.4e}, ki={ki:.4e}"
        )
        if env.reward_breakdown is True:
            print("Reward component breakdown:")

            for name, total in pos_component_totals.items():
                share = (total / pos_total_sum * 100) if pos_total_sum != 0 else 0
                print(f"  {name}: {total:.4f} ({share:.1f}%)")
            # j_error_tot = sum(env._reward_components["j_error"])
            # j_error_share = (
            #     (-j_error_tot / pos_total_sum * 100) if pos_total_sum != 0 else 0
            # )
            # print(f"  j_error: {j_error_tot:.4f} ({j_error_share:.1f}%)")
        print()  # Empty line for readability

    # Plot j evolution if requested
    if save_plot_as:
        agent.plot_j_evolution(filename=save_plot_as)
        print("plots saved !")

    return -cumulative_reward


def optimize(function, grid_search_space, bounds):
    """Optimize PID parameters using grid search followed by local optimization.

    Args:
        function: Objective function to minimize
        grid_search_space: Tuple of (Kp_vals, Ki_vals) arrays for grid search
        bounds: Bounds for local optimization

    Returns:
        scipy.optimize.OptimizeResult: Best optimization result
    """
    start_time = time.time()

    # coarse grid in bounds
    Kp_vals = grid_search_space[0]
    Ki_vals = grid_search_space[1]
    candidates = [(kp, ki) for kp in Kp_vals for ki in Ki_vals]

    # evaluate coarse grid
    scores = []
    grid_start_time = time.time()
    print("Starting grid search...")
    for x in candidates:
        scores.append((function(np.array(x)), x))
    scores.sort()  # ascending (minimization of negative reward)
    grid_time = time.time() - grid_start_time

    print(f"Grid search done in {grid_time:.2f}s. Top scores:")
    for val, (kp, ki) in scores[:5]:
        print(f"  Kp: {kp:.4e}, Ki: {ki:.4e}, Reward: {-val:.4f}")

    # start local optimization from top-N grid points
    local_start_time = time.time()
    best = None
    best_val = np.inf
    for val, x0 in scores[:5]:  # refine top-5 starting points
        res = minimize(
            function,
            np.array(x0),
            method="Powell",
            options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 50},
            bounds=bounds,
        )
        print("Optimization result:", res.x, res.fun)
        if res.fun < best_val:
            best_val = res.fun
            best = res
    local_time = time.time() - local_start_time
    total_time = time.time() - start_time

    print(f"Local optimization done in {local_time:.2f}s")
    print(f"Total optimization time: {total_time:.2f}s")
    print("Best solution:", best.x, "reward:", -best.fun)

    return best


if __name__ == "__main__":
    env = IterHybridEnv(render_mode=None, store_history=True, log_level="error")

    observation, _ = env.reset()

    # Example 1: Direct scipy.optimize.minimize with timing
    # opt_start_time = time.time()
    # res = minimize(
    #     lambda k: simulate(env, k),
    #     x0=[10, 10],
    #     method="Powell",
    #     options={
    #         "disp": True,  # Display convergence messages
    #     },
    #     bounds=[(0, 50), (0, 50)],
    # )
    # opt_time = time.time() - opt_start_time
    # kp_opt, ki_opt = res.x
    # print(f"Optimal Kp, Ki: {kp_opt:.4e}, {ki_opt:.4e}, Value: {-res.fun}")
    # print(f"Optimization completed in {opt_time:.2f}s")

    # Example 2: Grid search + local optimization with built-in timing
    # res = optimize(
    #     lambda k: simulate(env, k),
    #     grid_search_space=(
    #         np.linspace(0, 25, 5),  # Kp values
    #         np.linspace(0, 25, 5),  # Ki values
    #     ),
    #     bounds=[(0, 25), (0, 25)],
    # )

    # kp, ki = res.x

    kp, ki = 0.20176247, 19.09879356

    main_start_time = time.time()
    simulate(
        env,
        [kp, ki],
        save_plot_as="tmp/pid_optimization",
    )
    main_time = time.time() - main_start_time
    print(f"Main execution completed in {main_time:.2f}s")
    """
    Only with j_error:
        Optimization terminated successfully.
            Current function value: 4.379804
            Iterations: 2
            Function evaluations: 121
        Optimal Kp, Ki: 4.5682e+01, 4.1315e+00, Value: -4.3798039897847305

    + -1000 if crash:
    Optimization terminated successfully.
         Current function value: 6.528004
         Iterations: 2
         Function evaluations: 120
    Optimal Kp, Ki: 3.4669e+01, 2.3164e-01, Value: -6.528003919100931


    Total cumulative reward: 51.9981
    Reward component breakdown:
    fusion_gain: 51.0004 (98.1%)
    q_min: 7.4000 (14.2%)
    q_edge: 15.1000 (29.0%)
    j_error: -21.5023 (-41.4%)

    Iteration 2: kp=1.0840e+27, ki=1.1057e+00
    Optimization terminated successfully.
            Current function value: -51.998101
            Iterations: 2
            Function evaluations: 73
    Optimal Kp, Ki: 6.2250e+01, 1.0050e-01, Value: 51.99810094340189
    """
    env.save_gif_torax(
        filename="tmp/pid_optimized.gif",
        interval=250,
        frame_skip=5,
        config_plot="default",
        beginning=0,
        end=-1,
    )
