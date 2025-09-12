import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from gymtorax import IterHybridEnv

# Set up logger for this module
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 25})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

_NBI_W_TO_MA = 1 / 16e6

nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

r_nbi = 0.25
w_nbi = 0.25

eccd_power = {0: 0, 99: 0, 100: 20.0e6}


class IterHybridEnv(IterHybridEnv):  # noqa: D101
    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        self.reward_breakdown = True  # Enable reward breakdown tracking


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


class IterHybridAgent:  # noqa: D101
    """Agent for the ITER hybrid scenario.

    This agent produces a sequence of actions for the ITER hybrid scenario,
    ramping up plasma current and heating sources according to the scenario timeline.
    """

    def __init__(self, action_space):
        """Initialize the agent with the given action space."""
        self.action_space = action_space
        self.time = 0
        self.action_history = []

    def act(self, observation) -> dict:
        """Compute the next action based on the current observation and internal time.

        Returns:
            dict: Action dictionary for the environment.
        """
        action = {
            "Ip": [3e6],
            "NBI": [nbi_powers[0], r_nbi, w_nbi],
            "ECRH": [eccd_power[0], 0.35, 0.05],
        }

        if self.time == 98:
            action["ECRH"][0] = eccd_power[99]
            action["NBI"][0] = nbi_powers[1]

        if self.time >= 99:
            action["ECRH"][0] = eccd_power[100]
            action["NBI"][0] = nbi_powers[2]

        if self.time < 99:
            action["Ip"][0] = 3e6 + (self.time + 1) * (12.5e6 - 3e6) / 100
        else:
            action["Ip"][0] = 12.5e6

        self.time += 1

        self.action_history.append(action["Ip"][0] / 1e6)
        return action


def simulate(env: IterHybridEnv, agent, verbose=0):
    """Simulate the environment with given PID parameters and return a cost.

    Args:
        env: The environment to simulate
        agent: The agent to use for action selection
        verbose: Verbosity level (0: none, 1: total reward, 2: detailed breakdown)

    Returns:
        float: Negative cumulative reward (cost to minimize)
    """
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
        r_components = {
            name: sum(values) for name, values in env._reward_components.items()
        }
        pos_total_sum = sum(r_components.values())

        if verbose > 0:
            if hasattr(agent, "kp") and hasattr(agent, "ki"):
                text = f" with kp={agent.kp:.4e}, ki={agent.ki:.4e}"
            else:
                text = ""
            print(f"Total cumulative reward: {cumulative_reward:.4f}{text}")
        if env.reward_breakdown is True and verbose > 1:
            print("Reward component breakdown:")

            for name, total in r_components.items():
                share = (total / pos_total_sum * 100) if pos_total_sum != 0 else 0
                print(f"  {name}: {total:.4f} ({share:.1f}%)")

        # print()  # Empty line for readability

    return -cumulative_reward


def optimize(function, bounds, n_starts=5):
    """Optimize PID parameters using multiple evenly distributed starting points.

    Args:
        function: Objective function to minimize
        bounds: Bounds for optimization [(kp_min, kp_max), (ki_min, ki_max)]
        n_starts: Number of evenly distributed starting points (default: 5)

    Returns:
        scipy.optimize.OptimizeResult: Best optimization result
    """
    start_time = time.time()

    # Extract bounds
    kp_bounds, ki_bounds = bounds
    kp_min, kp_max = kp_bounds
    ki_min, ki_max = ki_bounds

    # Create evenly distributed initial conditions avoiding borders
    # Use n_starts+1 points and exclude the first and last (border) points
    kp_starts = np.linspace(kp_min, kp_max, n_starts + 2)[1:-1]
    ki_starts = np.linspace(ki_min, ki_max, n_starts + 2)[1:-1]

    # Create all combinations of starting points
    initial_conditions = []
    for kp in kp_starts:
        for ki in ki_starts:
            initial_conditions.append([kp, ki])

    print(
        f"Starting optimization with {len(initial_conditions)} evenly distributed initial conditions..."
    )

    # Run minimize from each starting point
    best = None
    best_val = np.inf
    results = []

    for i, (kp0, ki0) in enumerate(initial_conditions):
        print(
            f"Starting point {i + 1}/{len(initial_conditions)}: kp={kp0:.4e}, ki={ki0:.4e}"
        )

        res = minimize(
            function,
            np.array([kp0, ki0]),
            method="Powell",
            options={"xtol": 1e-3, "ftol": 1e-4},
            bounds=bounds,
        )

        results.append(res)
        print(
            f"  Result: kp={res.x[0]:.4e}, ki={res.x[1]:.4e}, objective={res.fun:.4f}"
        )

        if res.fun < best_val:
            best_val = res.fun
            best = res

    total_time = time.time() - start_time

    print(f"\nOptimization completed in {total_time:.2f}s")
    print(
        f"Best solution: kp={best.x[0]:.4e}, ki={best.x[1]:.4e}, reward={-best.fun:.4f}"
    )

    return best


def _plot_j_evolution(
    filename,
    action_history,
    j_target_history,
    j_actual_history,
    ip_reference,
):
    fig, ax = plt.subplots()

    j_target_ma = np.array(j_target_history) / 1e6
    j_actual_ma = np.array(j_actual_history) / 1e6

    # Plot j_target and j_actual
    ax.plot(
        j_actual_ma,
        "b-",
        label=r"$j_{actual}$",
        linewidth=2,
    )

    ax.plot(
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
    ax.legend(prop={"size": 20}, loc="upper left")

    fig.savefig(f"{filename}_error.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(
        np.array(action_history) / 1e6,
        linewidth=2,
        color="blue",
        label=r"$I_p$",
    )
    if ip_reference is not None:
        ax.plot(
            np.array(ip_reference),
            linewidth=1.5,
            label=r"$I_{p,ref}$",
            color="blue",
            alpha=0.6,
            linestyle="dotted",
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
    ax.legend(prop={"size": 20}, loc="upper left")

    fig.savefig(f"{filename}_ip.pdf", bbox_inches="tight")


if __name__ == "__main__":
    get_j_target = lambda t: 0.2e6 + 0.4e6 + 1.4e6 * t / 100  # noqa: E731

    env = IterHybridEnv(render_mode=None, store_history=True, log_level="critical")

    observation, _ = env.reset()

    def _simulate_pid(k):
        agent = PIDAgent(
            env.action_space,
            get_j_target,
            env.action_handler.get_actions()["Ip"].ramp_rate[0],
            kp=k[0],
            ki=k[1],
            kd=0.0,
        )

        return simulate(env, agent, verbose=1)

    # Run the main optimization sequence
    # res = optimize(
    #     lambda k: _simulate_pid(k),
    #     bounds=[(0, 50), (0, 50)],
    #     n_starts=3,
    # )
    # kp, ki = res.x

    kp, ki = 0.32521673, 30.9023307

    agent_pid = PIDAgent(
        env.action_space,
        get_j_target,
        env.action_handler.get_actions()["Ip"].ramp_rate[0],
        kp=kp,
        ki=ki,
        kd=0.0,
    )

    # Final simulation with optimized parameters and plotting
    simulate(
        env,
        agent_pid,
        verbose=2,
    )

    j_target_history = [get_j_target(t) for t in range(150)]

    # Save a gif of the final simulation
    # env.save_gif_torax(
    #     filename="tmp/pid_optimized.gif",
    #     interval=250,
    #     frame_skip=2,
    #     config_plot="default",
    #     beginning=0,
    #     end=-1,
    # )

    # Also run a simulation with the IterHybridAgent for comparison
    agent_classic = IterHybridAgent(env.action_space)

    simulate(
        env,
        agent_classic,
        verbose=2,
    )

    _plot_j_evolution(
        "tmp/pid_optimized",
        agent_pid.action_history,
        agent_pid.j_target_history,
        agent_pid.j_actual_history,
        agent_classic.action_history,
    )
