import numpy as np

from gymtorax import IterHybridEnv
from gymtorax.rendering.plots import main_prop_fig

_NBI_W_TO_MA = 1 / 16e6
W_to_Ne_ratio = 0

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

        return action


if __name__ == "__main__":
    env = IterHybridEnv(render_mode="human", store_history=True, fig=main_prop_fig)
    agent = IterHybridAgent(env.action_space)

    observation, _ = env.reset()
    terminated = False

    i = 0
    while not terminated:
        action = agent.act(observation)
        observation, _, terminated, _, _ = env.step(action)
        i += 1
        if i % 10 == 0:
            env.render()

    env.save_file("tmp/outputs_iter_torax.nc")

    env.save_gif_nc(
        nc_file="tmp/outputs_iter_torax.nc",
        filename="tmp/output_torax.gif",
        config_plot=main_prop_fig,
        interval=200,
        frame_skip=5,
    )
    # env.save_gif_torax(
    #     filename="tmp/source_output_torax.gif",
    #     config_plot=main_prop_fig,
    #     interval=200,
    #     frame_skip=5,
    #     )
