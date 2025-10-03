Other Modules
=============

This section provides documentation for additional modules in GymTorax:

.. contents::
    :local:
    :depth: 1

Logging System
--------------

Comprehensive logging configuration for debugging and monitoring.

.. automodule:: gymtorax.logger
   :members:
   :undoc-members:
   :show-inheritance:

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gymtorax.logger import setup_logging
   import logging

   # Configure comprehensive logging
   setup_logging(
       level=logging.DEBUG,
       log_file="simulation.log",
       suppress_external=True  # Quiet JAX/TORAX messages
   )

   # Environment will use configured logging
   env = gym.make('gymtorax/IterHybrid-v0', log_level="debug")

Custom Log Levels
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different log levels for different components
   setup_logging(level=logging.INFO)
   
   # Set specific loggers
   logging.getLogger("gymtorax.torax_wrapper").setLevel(logging.DEBUG)
   logging.getLogger("gymtorax.envs").setLevel(logging.WARNING)

Visualization and Rendering
---------------------------

Real-time visualization system with TORAX integration for plasma simulation monitoring and video recording.

.. automodule:: gymtorax.rendering
   :members:
   :undoc-members:
   :show-inheritance:

Real-time Visualization
~~~~~~~~~~~~~~~~~~~~~~~

The visualization system provides live plotting during simulation episodes with support for both interactive display and video recording:

.. code-block:: python

   import gymnasium as gym
   from gymnasium.wrappers import RecordVideo
   
   # Interactive visualization
   env = gym.make('gymtorax/IterHybrid-v0', render_mode="human")
   obs, info = env.reset()
   
   for _ in range(100):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       env.render()  # Live matplotlib display
       
       if terminated or truncated:
           break

TORAX Plot Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Uses standard TORAX plot configurations, or create custom configurations and pass them to the `plot_config` argument of the environment:

.. code-block:: python

   from gymtorax.rendering import Plotter, process_plot_config
   
   # Custom configuration for specific variables
   env = gym.make('gymtorax/IterHybrid-v0', 
                  render_mode="human",
                  plot_config="simple")

