Environment Design
===========================

This section describes the implementation of the environment design modules.
Some of them are mentioned in the :doc:`User Guide <../user_guide/index>`.

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
   :members: reset, step, close, render, save_file, save_gif, _define_action_space, _define_observation_space, _get_torax_config, _compute_reward
   :no-index:

Rewards
----------------
 
.. automodule:: gymtorax.rewards
    :members:
