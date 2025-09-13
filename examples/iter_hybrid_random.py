import cProfile

import numpy as np

from gymtorax import IterHybridEnv

_NBI_W_TO_MA = 1 / 16e6
W_to_Ne_ratio = 0

nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

r_nbi = 0.25
w_nbi = 0.25

eccd_power = {0: 0, 99: 0, 100: 20.0e6}


class IterHybridRandomAgent:  # noqa: D101
    """Agent for the ITER hybrid scenario.

    This agent produces random actions within the action space.
    """

    def __init__(self, action_space):
        """Initialize the agent with the given action space."""
        self.action_space = action_space

    def act(self, observation) -> dict:
        """Compute the next action based on the current observation and internal time.

        Returns:
            dict: Action dictionary for the environment.
        """
        action = self.action_space.sample()

        return action


if __name__ == "__main__":
    profiler = cProfile.Profile()

    env = IterHybridEnv(render_mode="human", store_history=True)
    env.reward_breakdown = True
    agent = IterHybridRandomAgent(env.action_space)


    observation, _ = env.reset()
    terminated = False

    terminated = False
    cumulative_reward = 0.0
    n = 0
    gamma = 1  # Discount factor for future rewards

    while not terminated:
        action = agent.act(observation)
        observation, reward, terminated, _, _ = env.step(action)

        cumulative_reward += reward * (gamma**n)
        n += 1
    
    print(f"Total cumulative reward: {cumulative_reward:.4f}")

    # Calculate and log reward component shares
    if hasattr(env, "_reward_components"):
        r_components = {
            name: sum(values) for name, values in env._reward_components.items()
        }
        pos_total_sum = sum(r_components.values())

        
        print(f"Total cumulative reward: {cumulative_reward:.4f}")
        if env.reward_breakdown is True:
            print("Reward component breakdown:")

            for name, total in r_components.items():
                share = (total / pos_total_sum * 100) if pos_total_sum != 0 else 0
                print(f"  {name}: {total:.4f} ({share:.1f}%)")

    # env.save_gif_torax("tmp/iter_hybrid_random.gif", frame_skip=2, config_plot="default")
