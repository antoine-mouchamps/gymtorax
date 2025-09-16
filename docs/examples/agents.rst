Agents
==========
Three simple agents are included in this example:

- **Open-loop reference agent** – reproduces the original TORAX trajectories. This validates 
  that Gym-TORAX is consistent with the reference simulator.

- **PI controller agent** – regulates the plasma current using a Proportional-Integral 
  controller. The controller gains are optimized to maximize the cumulative reward. 
  Heating actions (NBI, ECRH) follow the reference trajectories.

- **Random agent** – selects actions uniformly at random within the allowed ranges. 
  This provides a naive baseline for comparison.

Here is the code for the open-loop agent as illustration:

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

