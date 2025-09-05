Environment Design
===========================

This section describes the implementation of the environment design modules.

Four main components are covered:

.. contents::
    :local:

action_handler
----------------

.. automodule:: gymtorax.action_handler
   :members:

observation_handler
--------------------

.. automodule:: gymtorax.observation_handler
   :members:

base_env
----------------

.. automodule:: gymtorax.envs.base_env

.. autoclass:: gymtorax.envs.base_env.BaseEnv
   :members: reset, step, close, render, save_file, save_gif, _define_actions, _define_observation, _define_torax_config, _define_reward

.. Rewards
.. ----------------
.. 
.. .. automodule:: gymtorax.rewards
..    :members:
