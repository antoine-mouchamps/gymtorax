GymTorax Documentation
======================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/pypi/v/gymtorax.svg
   :target: https://pypi.org/project/gymtorax/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/antoine-mouchamps/gymtorax/blob/main/LICENSE
   :alt: License

**A Gymnasium environment for reinforcement learning in tokamak plasma control**

Gym-TORAX is a Python package that provides reinforcement learning (RL) 
environments for plasma control, built on top of the `TORAX plasma simulator <https://torax.readthedocs.io/en/v1.0.3/index.html>`_.

Its purpose is to bridge the gap between plasma physics simulation and RL research:
   - For RL users, it exposes ready-to-use environments following the `Gymnasium API <https://gymnasium.farama.org/index.html>`_, abstracting away the plasma physics.
   - For plasma physicists, it provides a framework to design and customize new control tasks, making it easy to test RL algorithms in different operational scenarios.

Key Features
------------

* **Gymnasium Integration**: Standard RL environment interface compatible with the Gymnasium ecosystem
* **TORAX Physics**: 1D transport equations solved with TORAX plasma physics models
* **Configurable Environment**: Flexible action spaces, observation spaces, and reward functions

Getting Started
---------------

**Installation**

.. code-block:: bash

   pip install gymtorax

**Basic Usage**

.. code-block:: python

   import gymnasium as gym
   import gymtorax

   # Create environment
   env = gym.make('gymtorax/IterHybrid-v0')
   
   # Reset and run with a random agent
   observation, info = env.reset()
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   about_gymtorax
   installation
.. toctree::
   :maxdepth: 2
   :caption: Additional Resources
   :hidden:

   user_guide/index
   developer_guide/index
   examples/index
   citing