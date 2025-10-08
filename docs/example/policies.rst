Policies
==========
Three policies included in this example:

- **Open-loop reference policy** :math:`\pi_{OL}` – uses a predetermined set of actions that directly follows the action trajectories of the initial scenario given in TORAX.

- **Random policy** :math:`\pi_{R}` – selects the actions uniformly at random.

- **PI controller policy** :math:`\pi_{PI}` – controls the total current action using a PI controller and uses the same predetermined trajectories as the open-loop policy for the last two actions, NBI and ECRH. The PI controller is used to follow a prescribed linear increase (from :math:`0.6\,\mathrm{MA/m^2}` to :math:`2\,\mathrm{MA/m^2}`) of the current density at the center of the plasma during the ramp-up phase of 100 seconds. Once the ramp-up has been performed, action values are kept constant from the last action given by the PI controller until the end of the episode (for the last 49 seconds). The proportional :math:`k_p` and integral :math:`k_i` gains of the PI controller are optimized to maximize the expected return :math:`J(\pi)`. Carrying out the optimization, we obtain the following estimations for the optimal parameters: :math:`{\hat{k}_p^*=0.700}` and :math:`{\hat{k}_i^*=34.257}`. More information about the optimization procedure are given in the article presenting Gym-TORAX.

Here is the implementation of the open-loop policy as illustration:

.. code-block:: python

    import numpy as np

    _NBI_W_TO_MA = 1 / 16e6

    nbi_powers = np.array([0, 0, 33e6])
    nbi_cd = nbi_powers * _NBI_W_TO_MA

    r_nbi = 0.25
    w_nbi = 0.25

    eccd_power = {0: 0, 99: 0, 100: 20.0e6}

    class IterHybridAgent: 
        def __init__(self, action_space):
            self.action_space = action_space
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

