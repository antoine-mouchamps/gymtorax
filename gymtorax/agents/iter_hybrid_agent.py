import numpy as np

_NBI_W_TO_MA = 1 / 16e6

nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

r_nbi = 0.25
w_nbi = 0.25

eccd_power = {0: 0, 99: 0, 100: 20.0e6}


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
