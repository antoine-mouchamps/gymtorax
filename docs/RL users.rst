RL Users
==============
The **Gym-TORAX** package provides a flexible framework for developing reinforcement 
learning (RL) agents in plasma physics, using the well-known Gymnasium interface. The 
core implementation can be found in `gymtorax/envs/base_env` or :doc:`here <Environment Design>` 
in the documentation. The main methods 

- :py:meth:`reset()<gymtorax.envs.base_env.BaseEnv.reset>`: reset the environment to its initial state for a new episode.
- :py:meth:`step(action)<gymtorax.envs.base_env.BaseEnv.step>`: execute one environment step with the given action. It returns the next observation,
  reward, done flag, and additional info.
- :py:meth:`render()<gymtorax.envs.base_env.BaseEnv.render>`: render the current environment state following Gymnasium convention.
- :py:meth:`close()<gymtorax.envs.base_env.BaseEnv.close>`: clean up environment resources.

Other methods are available to analyse outputs, save data, etc.

- :py:meth:`save_file(file_name)<gymtorax.envs.base_env.BaseEnv.save_file>`: save the simulation output data to a file.
- :py:meth:`save_gif(filename, interval, frame_step)<gymtorax.envs.base_env.BaseEnv.save_gif>`: save the data as a GIF file.

Here is a simple example of how to use the Gym-TORAX package for reinforcement learning tasks:

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