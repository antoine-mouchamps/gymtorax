PI Controller
======================

This example illustrates how Gym-TORAX can be used for control tasks with a 
simple Proportional–Integral (PI) controller.

In this setup, the plasma current (:math:`I_p`) is controlled by a PI controller with 
proportional and integral gains (:math:`k_p` and :math:`k_i`). The other actuators 
(NBI and ECRH) follow the same trajectories as in the reference scenario. The controller 
is applied during the ramp-up phase, where the central current density is required to 
increase linearly from 0.6 MA/m² to 2.0 MA/m². After the ramp-up, the actions are kept 
constant until the end of the simulation.

To tune the PI gains, we optimized :math:`k_p` and :math:`k_i` to maximize the 
cumulative reward defined in the environment. The optimized controller improves the 
total reward compared to the reference scenario, providing a useful baseline for 
reinforcement learning algorithms.

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
   * - Reward PI controller
     - :math:`R_{PI}`
     - 4.7829
   * - Reward reference scenario
     - :math:`R_{ref}`
     - 2.6668

The figure below shows the resulting plasma current evolution and the tracking error 
of the controller with respect to the prescribed ramp-up trajectory.

.. grid:: 1

    .. grid-item::
        .. figure:: Images/pid_control_ip.jpg
            :align: center
            :width: 100%

            Total current evolution under PI control

Results of the PI controller: the optimized gains achieve a better reward than the reference scenario, demonstrating Gym-TORAX can be used as a testbed for control strategies.
