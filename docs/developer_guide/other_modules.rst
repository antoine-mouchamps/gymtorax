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
       logfile="simulation.log",
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

Tools for plotting and visualizing plasma simulations.

WIP