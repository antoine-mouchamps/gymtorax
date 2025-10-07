Results
===========

The three agents were evaluated and compared in terms of expected return:

.. list-table:: Parameters of the PI controller and comparison of rewards
   :align: center

   * - Parameter
     - Symbol
     - Value
   * - Proportional
     - :math:`k_p`
     - 0.032
   * - Integral
     - :math:`k_i`
     - 30.090
   * - Reward reference scenario
     - :math:`R_{ref}`
     - 2.6668
   * - Reward PI controller
     - :math:`R_{PI}`
     - 4.7829
   * - Reward random agent
     - :math:`R_{rand}`
     - ...

**Reference agent** reproduces the exact TORAX outputs, confirming that Gym-TORAX 
preserves the dynamics of the original simulator.
To illustrate this agreement, we compare current density profiles at four representative times 
(50s, 99s, 105s, and 150s), spanning both the ramp-up and the nominal phases of the scenario:

.. grid:: 2

    .. grid-item::
        .. figure:: ../Images/comparison_50.jpg
            :align: center

            : t = 50s

    .. grid-item::
        .. figure:: ../Images/comparison_99.jpg
            :align: center

            : t = 99s

    .. grid-item::
        .. figure:: ../Images/comparison_105.jpg
            :align: center

            : t = 105s

    .. grid-item::
        .. figure:: ../Images/comparison_150.jpg
            :align: center

            : t = 150s

*Snapshots of current density at different times (native TORAX: dashed lines, 
Gym-TORAX: solid lines).*

**PI controller** improves the reward compared to the reference case. The optimized 
gains produce a slight overshoot in the plasma current during the ramp-up phase, 
a behavior consistent with experimental observations, showing that the environment 
captures realistic control dynamics.

.. grid:: 1

    .. grid-item::
        .. figure:: ../Images/pid_optimized_ip.jpg
            :align: center
            :width: 70%

            Total current evolution under PI control

**Random agent** (not shown) achieves very low reward, as expected.


These results show that Gym-TORAX can accommodate different types of agents, from 
open-loop reproduction to closed-loop control, while preserving the physical fidelity 
of the underlying simulator. The PI controller provides a simple baseline against which 
future reinforcement learning or more advanced control algorithms can be compared.