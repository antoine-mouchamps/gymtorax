Physical validation
====================
To validate that Gym-TORAX reproduces TORAX faithfully, we implemented an agent 
with a predetermined policy. This agent directly follows the action trajectories 
from the TORAX reference ITER hybrid scenario.

.. code-block:: python

    class IterHybridAgent(BaseAgent): 
        def __init__(self, action_space):
            super().__init__(action_space=action_space)
            self.time = 0

        def act(self, observation) -> dict:
            action = {
                "Ip": [3e6],
                "NBI": [nbi_powers[0], nbi_cd[0], r_nbi, w_nbi],
                "ECRH": [eccd_power[0], 0.35, 0.05],
            }

            if self.time == 98:
                action["ECRH"][0] = eccd_power[99]
                action["NBI"][0] = nbi_powers[1]
                action["NBI"][1] = nbi_cd[1]

            if self.time >= 99:
                action["ECRH"][0] = eccd_power[100]
                action["NBI"][0] = nbi_powers[2]
                action["NBI"][1] = nbi_cd[2]

            if self.time < 99:
                action["Ip"][0] = 3e6 + (self.time + 1) * (12.5e6 - 3e6) / 100
            else:
                action["Ip"][0] = 12.5e6

            self.time += 1

            return action



Comparison between Gym-TORAX and native TORAX shows identical plasma evolution for all 
state variables. This validates the correctness of the wrapper implementation. To illustrate 
this agreement, we compare current density profiles at four representative times 
(50s, 99s, 105s, and 150s), spanning both the ramp-up and the nominal phases of the scenario:

.. grid:: 2

    .. grid-item::
        .. figure:: Images/comparison_50.jpg
            :align: center

            : t = 50s

    .. grid-item::
        .. figure:: Images/comparison_99.jpg
            :align: center

            : t = 99s

    .. grid-item::
        .. figure:: Images/comparison_105.jpg
            :align: center

            : t = 105s

    .. grid-item::
        .. figure:: Images/comparison_150.jpg
            :align: center

            : t = 150s

*Snapshots of current density at different times (native TORAX: dashed lines, 
Gym-TORAX: solid lines).*

Our complete code is available
`here <https://github.com/antoine-mouchamps/gymtorax/blob/main/examples/iter_hybrid.py>`_.