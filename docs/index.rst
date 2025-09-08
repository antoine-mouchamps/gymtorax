Welcome page
================

**Gym-TORAX** is a Python package that provides reinforcement learning (RL) 
environments for plasma control, built on top of the `TORAX plasma simulator <https://torax.readthedocs.io/en/v1.0.3/index.html>`_.

Its purpose is to bridge the gap between plasma physics simulation and RL research:

- For RL users, it exposes ready-to-use environments following the `Gymnasium API <https://gymnasium.farama.org/index.html>`_, abstracting away the plasma physics.

- For plasma physicists, it provides tools to design and customize new control tasks, making it easy to test RL algorithms in different operational scenarios.

Key Features:

- Gymnasium-compliant environments: seamless integration with popular RL libraries.

- Plasma dynamics powered by TORAX: realistic 1D transport equations solved efficiently with JAX.

- Configurable reward functions: combine predefined terms to target specific control goals.

- Flexible environment design: easily define action spaces, observation spaces, and simulation parameters.

Gym-TORAX lowers the entry barrier for RL researchers to explore plasma control problems, while giving physicists a straightforward way to create environments tailored to their needs.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   About GymTorax
   Installation
   User guide
   Developer guide
   Examples
   Citing