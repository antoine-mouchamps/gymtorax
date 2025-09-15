import numpy as np

from gymtorax import IterHybridEnv, RandomAgent


def simulation_run(env, agent):
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

    return cumulative_reward

if __name__ == "__main__":
    env = IterHybridEnv(log_level="critical")
    agent = RandomAgent(env.action_space)

    rewards = []
    while True:
        rewards.append(simulation_run(env, agent))
        average_reward = np.mean(rewards)
        std_var_reward = np.std(rewards)
        print(
            f"Run #{len(rewards)} Average reward: {average_reward:.4f} +/- {std_var_reward:.4f}"
        )
