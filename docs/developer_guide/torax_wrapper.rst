TORAX Wrapper
=============

The TORAX wrapper is a critical component that bridges between the Gymnasium 
environment interface and the TORAX plasma physics simulation engine. It handles
state management, configuration, and provides a clean interface for reinforcement
learning interactions.

.. contents::
    :local:

torax_app
---------

The main application class that orchestrates TORAX simulations.

.. automodule:: gymtorax.torax_wrapper.torax_app

config_loader
-------------

Configuration system for managing TORAX physics parameters and simulation settings.

.. automodule:: gymtorax.torax_wrapper.config_loader
