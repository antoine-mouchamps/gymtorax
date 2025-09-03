import numpy as np
from iter_hybrid_pid import IterHybridEnvPid
from scipy.optimize import minimize


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
