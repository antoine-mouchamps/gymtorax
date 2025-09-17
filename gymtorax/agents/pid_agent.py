import numpy as np

_NBI_W_TO_MA = 1 / 16e6

nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

r_nbi = 0.25
w_nbi = 0.25

eccd_power = {0: 0, 99: 0, 100: 20.0e6}


class PIDAgent:  # noqa: D101
    def __init__(self, action_space, get_j_target, ramp_rate, kp, ki, kd):  # noqa: D107
        self.action_space = action_space
        self.get_j_target = get_j_target
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

    def act(self, observation) -> dict:  # noqa: D102
        j_center = observation["profiles"]["j_total"][0]
        j_target = self.get_j_target(self.time)

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

            # Apply ramp rate limiting (0.2 MA/s = 0.2e6 A/s)
            max_ramp_rate = self.ramp_rate
            max_change = max_ramp_rate * self.dt  # Maximum change per time step

            is_ramp_limited = False
            if self.time > 0:  # Only apply ramp rate limiting after first step
                ip_change = ip_desired - self.ip_controlled
                if abs(ip_change) > max_change:
                    is_ramp_limited = True
                    # Limit the change to the maximum ramp rate
                    ip_ramp_limited = (
                        self.ip_controlled + np.sign(ip_change) * max_change
                    )
                else:
                    ip_ramp_limited = ip_desired
            else:
                ip_ramp_limited = ip_desired  # No ramp limit on first step

            # Then apply physical power limits
            ip_final = np.clip(ip_ramp_limited, self.ip_min, self.ip_max)

            # Check what type of limiting is occurring
            is_power_limited = ip_final != ip_ramp_limited
            if is_power_limited:
                ip_actual_change = ip_final - self.ip_controlled
                if is_ramp_limited is True and abs(ip_actual_change) < max_change:
                    is_ramp_limited = False

            # Anti-windup: only update integral if not limited
            if self.anti_windup_enabled and (is_ramp_limited or is_power_limited):
                pass
            else:
                # Standard integral update (no anti-windup or not limited)
                self.error_integral += error * self.dt

            # Update the controlled value
            self.ip_controlled = ip_final

            # Store current error for next derivative calculation
            self.previous_error = error

        action = {
            "Ip": [self.ip_controlled],
            "NBI": [nbi_powers[0], r_nbi, w_nbi],
            "ECRH": [eccd_power[0], 0.35, 0.05],
        }

        if self.time == 98:
            action["ECRH"][0] = eccd_power[99]
            action["NBI"][0] = nbi_powers[1]

        if self.time >= 99:
            action["ECRH"][0] = eccd_power[100]
            action["NBI"][0] = nbi_powers[2]

        self.time += 1
        self.action_history.append(self.ip_controlled)

        return action
