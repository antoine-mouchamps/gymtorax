About Gym-TORAX
=================

Gym-TORAX is a Python package that turns the TORAX
plasma simulator into a set of reinforcement learning (RL) environments. 
Its main purpose is to connect two communities:

- RL users, who can train agents on realistic plasma control problems without 
  needing to understand the physics in detail.

- Plasma physicists, who can design custom environments tailored to their research 
  questions and evaluate RL control strategies in different scenarios.

TORAX in a Nutshell
--------------------

TORAX is an open-source plasma simulator based on 1D transport equations. 
It models the time evolution of key plasma quantities such as temperatures, densities, 
magnetic flux, and derived performance metrics (e.g. safety factor q, fusion gain Q).

In its native form, TORAX is an open-loop simulator: input parameters such as loop voltage, 
plasma current, or heating sources are predefined for the entire simulation, and the plasma 
state is updated accordingly.

Closing the Loop with Gym-TORAX
--------------------------------

Gym-TORAX transforms TORAX into a closed-loop control environment.

This is achieved through a two-level discretization:

- RL cycle → the agent observes the plasma, receives a reward, and selects an action.

- TORAX simulation → the chosen action is applied to the simulator, which evolves the 
  plasma state over multiple small time steps before returning the updated state.

This separation allows the physics to be simulated at fine resolution while the RL 
agent interacts at a higher-level control frequency.

The Environment Design
--------------------------

Every Gym-TORAX environment is built on top of a BaseEnv class. To create a new 
environment, four aspects must be specified:

- TORAX configuration → time discretization, physical models, geometry, and initial 
  conditions.

- Action space → which TORAX control variables are available to the agent (e.g. loop 
  voltage, heating powers).

- Observation space → which plasma state variables or derived metrics are provided to the agent.

- Reward function → combination of predefined reward terms to target control objectives 
  (stability, performance, etc.).

Helper classes (Action, Observation, Reward) simplify this process, so that new environments can 
be created with minimal boilerplate.

Beyond the Basics
-------------------

Gym-TORAX is more than just a Gymnasium wrapper:

- It abstracts away plasma physics for RL users, who can use the environments like any other Gymnasium task.

- It empowers physicists to explore advanced control strategies by creating environments with different assumptions or operating scenarios (steady-state, ramp-up, ramp-down, etc.).

- It includes visualization tools such as a real-time plotter and GIF exporter to track how the plasma evolves during an episode.