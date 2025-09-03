import numpy as np
from scipy.optimize import minimize

import gymtorax.action_handler as ah
import gymtorax.observation_handler as oh

import gymtorax.rendering.visualization as viz
import gymtorax.rewards as rw
from gymtorax import IterHybridEnv


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

    def act(self, observation, j_target) -> dict:
        if self.time >= 100:
            # keep the same self.ip_controlled after 100s
            pass

        else:
            j_center = observation["profiles"]["j_total"][0]

            # Calculate PID error (desired - actual)
            error = j_target - j_center

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
            # print(
            #     f"[PID DEBUG] t={self.time:3d}  j_obj={j_target:.3e}  j={j_center:.3e}  error={error:.3e}"
            # )
            # print(
            #     f"            P={p_term:.3e}  I={i_term:.3e}  D={d_term:.3e}  PID_out={pid_output:.3e}  I_int={self.error_integral:.3e}"
            # )
            # print(
            #     f"            Ip_ctrl={self.ip_controlled:.3e}  Ip_des={ip_desired:.3e}\n"
            # )

            # Then apply physical power limits
            ip_final = np.clip(ip_desired, self.ip_min, self.ip_max)

            # Check what type of limiting is occurring
            is_power_limited = ip_final != ip_desired

            # Anti-windup: only update integral if not limited, or if error would help
            if self.anti_windup_enabled and is_power_limited:
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

        return action


class IterHybridEnvPid(IterHybridEnv):  # noqa: D101
    def __init__(self, render_mode, fig=None, store_state_history=False):  # noqa: D107
        super().__init__(
            render_mode=render_mode,
            log_level="debug",
            fig=fig,
            store_state_history=store_state_history,
        )

    @property
    def _define_actions(self):  # noqa: D102
        actions = [ah.IpAction()]

        return actions

    def define_reward(self, state, next_state, action):
        """Compute the reward. The higher the reward, the more performance and stability of the plasma.

        The reward is a weighted sum of several factors:
        - Fusion gain Q: we want to maximize it.
        - Beta_N: we want to be as close as possible to the Troyon limit, but not exceed it too much.
        - Energy confinement time tau_E: we want to maximize it.
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
        Q = rw.get_fusion_gain(next_state)
        beta_N = rw.get_beta_N(next_state)
        tau_E = rw.get_tau_E(next_state)
        q_profile = rw.get_q_profile(next_state)
        q_min = rw.get_q_min(next_state)
        q_95 = rw.get_q95(next_state)
        s_profile = rw.get_s_profile(next_state)
        j_center = rw.get_j_profile(next_state)[0]

        # Customize weights and sigma as needed
        weight_list = [1, 1, 1, 1, 1, 1, 1]
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
                / (next_state["scalars"]["a_minor"] * next_state["scalars"]["B_0"])
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

        def j_error():
            """Compute the error between the actual and ideal current density.

            The ideal current density is a linear function of time.

            Returns:
                float: The error between the actual and ideal current density.
            """
            j_ideal = self._j_objectif()
            return abs(j_center - j_ideal)

        return (
            weight_list[0] * Q
            + weight_list[1] * gaussian_beta()
            + weight_list[2] * tau_E
            + weight_list[3] * q_min_function()
            + weight_list[4] * q_edge_function()
            + weight_list[5] * s_function()
            - weight_list[6] * j_error()
        )

    def _j_objectif(self):
        """Compute the objective function for the current density.

        Returns:
            float: The objective function value.
        """
        if self.current_time > 100:
            return 2e6
        else:
            return 1e6 * 1.5 / 100 * self.current_time + 0.4e6


def simulate(env: IterHybridEnvPid, k):
    """Simulate the environment with given PID parameters and return a cost."""
    kp, ki = k
    print(k)
    agent = PIDAgent(env.action_space, kp=kp, ki=ki, kd=0.0)

    observation, _ = env.reset()
    terminated = False
    cumulative_reward = 0.0
    n = 0
    gamma = 0.99  # Discount factor for future rewards

    while not terminated:
        action = agent.act(observation, env._j_objectif())
        observation, reward, terminated, _, _ = env.step(action)

        cumulative_reward += reward * (gamma**n)
        n += 1

    return -cumulative_reward


if __name__ == "__main__":
    from gymtorax.rendering.plots import main_prop_fig

    env = IterHybridEnvPid(
        render_mode=None, fig=main_prop_fig, store_state_history=True
    )

    observation, _ = env.reset()

    iteration = [0]  # Use a mutable object to track iteration count

    def _print_callback(xk):
        iteration[0] += 1
        kp, ki = np.exp(xk)
        print(f"Iteration {iteration[0]}: kp={kp:.4e}, ki={ki:.4e}")

    def simulate_log(env, k):
        kp, ki = np.exp(k)
        return simulate(env, [kp, ki])

    res = minimize(
        lambda k: simulate_log(env, k),
        x0=[1.0, -1000],
        method="Powell",
        callback=_print_callback,
    )
    kp_opt, ki_opt = np.exp(res.x)
    print(f"Optimal Kp, Ki: {kp_opt:.4e}, {ki_opt:.4e}, Value: {-res.fun}")
    optimal_agent = PIDAgent(env.action_space, kp=kp_opt, ki=ki_opt, kd=0.0)

    # optimal_agent = PIDAgent(env.action_space, kp=1e1, ki=0e4, kd=0.0)

    terminated = False
    observation, _ = env.reset()
    while not terminated:
        action = optimal_agent.act(observation, env._j_objectif())
        observation, reward, terminated, _, _ = env.step(action)
    env.save_gif_torax(
        filename="tmp/pid_optimized.gif",
        interval=250,
        frame_skip=2,
        config_plot="default",
        beginning=0,
        end=-1,
    )
