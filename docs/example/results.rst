Results
===========

The expected return obtained for each policy is given in :numref:`tab:results`.

.. list-table:: Expected return of the three studied policies, using a discount factor :math:`\gamma=1`.
   :align: center
   :width: 50%
   :name: tab:results

   * - Policy
     - .. container:: right-align

         Expected Return
   * - :math:`\pi_{OL}`
     - .. container:: right-align

         3.40
   * - :math:`\pi_{R}`
     - .. container:: right-align

         -10.79
   * - :math:`\pi_{PI}`
     - .. container:: right-align

         3.79

The open-loop policy yields an average return of $3.40$. As expected, the random policy performs worse, with an average return of $-10.79$. The best-performing policy is the PI controller-based one, with an average return of $3.79$. This result is an improvement over the reference scenario and can serve as a baseline for more sophisticated policies.

A representation of an action (total current) trajectory for each policy is given in :numref:`pid_optimized_ip`. This figure shows the erratic evolution of the total current of the random policy, which is somewhat mitigated by the ramp-rate constraints imposed in the environment. Regarding the PI policy, the trajectory of the current increases steadily and levels off at :math:`15\,\mathrm{MA}`, the maximum value allowable in the environment. This behavior is consistent with the fact that higher values of total current can generally be associated with improved confinement and overall better performance.

.. grid:: 1

    .. grid-item::
        .. figure:: ../Images/pid_optimized_ip.jpg
            :align: center
            :width: 50%
            :name: pid_optimized_ip

            Comparison of one action (total current) trajectory for each policy.


:numref:`pid_optimized_error` represents the target evolution of the PI controller and the action taken by the PI controller-based policy. Note that the parameters were optimized to maximize the expected return, rather than having actions close to the target, which can be observed in the figure.

.. grid:: 1

    .. grid-item::
        .. figure:: ../Images/pid_optimized_error.jpg
            :align: center
            :width: 50%
            :name: pid_optimized_error

            Evolution of the current density with respect to the target.
