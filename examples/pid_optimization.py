import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gymtorax.action_handler as ah
import gymtorax.observation_handler as oh
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
    def __init__(self, action_space, ramp_rate, kp, ki, kd):
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
        self.ip_min = action_space.spaces["Ip"].low[0]  # Minimum Ip current: 0 MA
        self.ip_max = action_space.spaces["Ip"].high[0]  # Maximum Ip power: 15 MA
        self.ramp_rate = ramp_rate  # Ramp rate limit in A/s

        # Tracking variables for plotting
        self.j_target_history = []
        self.j_actual_history = []
        self.time_history = []
        self.error_history = []
        self.action_history = []

    def act(self, observation) -> dict:
        j_center = observation["profiles"]["j_total"][0]
        j_target = 0.2e6 + 0.4e6 + 1.4e6 * self.time / 100

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

            # Then apply physical power limits
            ip_final = np.clip(ip_desired, self.ip_min, self.ip_max)

            # Apply ramp rate limiting (0.2 MA/s = 0.2e6 A/s)
            max_ramp_rate = self.ramp_rate
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

        _NBI_W_TO_MA = 1 / 16e6

        nbi_powers = np.array([0, 0, 33e6])
        nbi_cd = nbi_powers * _NBI_W_TO_MA

        r_nbi = 0.25
        w_nbi = 0.25

        eccd_power = {0: 0, 99: 0, 100: 20.0e6}

        action = {
            "Ip": [self.ip_controlled],
            "NBI": [nbi_powers[0], nbi_cd[0], r_nbi, w_nbi],
            "ECRH": [eccd_power[0], 0.35, 0.05],
        }

        if self.time == 98:
            action["ECRH"][0] = eccd_power[99]
            action["NBI"][0] = nbi_powers[1]
            action["NBI"][1] = nbi_cd[1]

        if self.time >= 99:
            action["ECRH"][0] = eccd_power[100]
            action["NBI"][0] = nbi_powers[2]
            action["NBI"][1] = nbi_cd[2]

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

        # Add vertical line at t=100 with LH transition text
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


def simulate(env: IterHybridEnv, k, save_plot_as=None):
    """Simulate the environment with given PID parameters and return a cost.

    Args:
        env: The environment to simulate
        k: PID parameters [kp, ki]
        save_plot_as: Filename to save the plot (if plotting is enabled)

    Returns:
        float: Negative cumulative reward (cost to minimize)
    """
    kp, ki = k

    agent = PIDAgent(
        env.action_space,
        env.action_handler.get_actions()["Ip"].ramp_rate[0],
        kp=kp,
        ki=ki,
        kd=0.0,
    )

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
        action = agent.act(observation)
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

    env.save_gif_torax(
        filename="tmp/pid_optimized.gif",
        interval=250,
        frame_skip=5,
        config_plot="default",
        beginning=0,
        end=-1,
    )
