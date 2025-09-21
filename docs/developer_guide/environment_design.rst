Environment Design
==================

This section describes the core environment components that make up GymTorax:
environments, actions, observations, and rewards. These are the building blocks
for creating plasma control tasks.

.. contents::
    :local:
    :depth: 2

Base Environment
----------------

.. automodule:: gymtorax.envs.base_env

Action Handling
---------------

The action handling system defines what parameters the RL agent can control
and how these map to TORAX configuration updates.

Action Handler
~~~~~~~~~~~~~~

.. autoclass:: gymtorax.action_handler.ActionHandler

Base Action Class
~~~~~~~~~~~~~~~~~

.. autoclass:: gymtorax.action_handler.Action

Concrete Actions
~~~~~~~~~~~~~~~~

.. autoclass:: gymtorax.action_handler.IpAction

.. autoclass:: gymtorax.action_handler.VloopAction

.. autoclass:: gymtorax.action_handler.EcrhAction

.. autoclass:: gymtorax.action_handler.NbiAction

Observation Handling
--------------------

The observation system extracts relevant plasma state information and formats
it for RL agents.

Base Observation Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gymtorax.observation_handler.Observation

Concrete Observations
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gymtorax.observation_handler.AllObservation

Reward Functions
----------------

Reward functions define the control objectives and translate plasma performance
into scalar signals for RL training.

.. automodule:: gymtorax.rewards
