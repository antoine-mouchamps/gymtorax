RL Users
==============

This section is intended for users who want to leverage the Gym-TORAX package for 
reinforcement learning tasks. It provides an overview of how to interact with 
the environments, including key methods and a simple example.

.. automethod:: gymtorax.envs.base_env.BaseEnv.reset
   :noindex:

.. automethod:: gymtorax.envs.base_env.BaseEnv.step
   :noindex:

.. automethod:: gymtorax.envs.base_env.BaseEnv.render
   :noindex:

.. automethod:: gymtorax.envs.base_env.BaseEnv.close
   :noindex:

.. automethod:: gymtorax.envs.base_env.BaseEnv.save_file
   :noindex:

.. automethod:: gymtorax.envs.base_env.BaseEnv.save_gif
   :noindex:


Here is a simple example of how to use the Gym-TORAX package for reinforcement 
learning applications:

.. code-block:: python

    import gymtorax
    env = gymtorax.make("basic_env")
    agent = YourRLAgent(env.action_space, env.observation_space)
    obs = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(obs)  # Get action from your RL agent
        obs, reward, terminated, truncated, info = env.step(action)
        agent.learn(obs, reward)  # Update your RL agent with the new observation and reward
        env.render()  # Optional: render the environment

    env.save_file("output_data")
    env.save_gif("simulation.gif")
    env.close()
