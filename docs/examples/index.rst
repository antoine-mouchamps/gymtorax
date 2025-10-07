Examples
=========
Example

To illustrate how Gym-TORAX can be used, we provide an example in which three different 
policies are used in the IterHybridEnv environment. The three policies are compared based on the expected return :math:`J(\pi)` given by

.. math::
   \begin{equation}
   J(\pi) = \underset{\pi}{\mathbb{E}} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right] \quad .
   \end{equation}

This example is organized into three parts:

1. :doc:`a description of the environment <iter_env>`,

2. :doc:`the agents and their control strategies <agents>`,

3. :doc:`the results obtained <results>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   iter_env
   agents
   results
   config_example