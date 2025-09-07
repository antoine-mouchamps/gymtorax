Physical validation
====================
To verify that Gym-TORAX does not alter the underlying physics, we reproduced 
the ITER hybrid ramp-up scenario. This configuration is one of the reference cases.
It is used in the `tutorial <https://torax.readthedocs.io/en/v1.0.3/tutorials.html>`_ (see [Citrin 2010]).

- Scenario: a 100 s ramp-up phase, followed by a 50 s nominal phase (0 – 150 s total).

- Control parameters: plasma current :math:`I_p`, neutral beam injection (NBI), and electron 
  cyclotron resonance heating (ECRH).

- Setup: we created a custom environment including three actions hereabove and the config file 
  from the TORAX tutorial. An agent was programmed with a predetermined policy reproducing 
  the input trajectories from the native TORAX scenario. The environment was stepped every 
  second, from 0 s to 150 s.

.. code-block:: python

    import gymtorax.action_handler as ah
    import gymtorax.observation_handler as oh
    from gymtorax.envs.base_env import BaseEnv

    class IterHybridEnv(BaseEnv):
        def __init__(self, render_mode, fig = None, store_state_history = False):
            super().__init__(
                render_mode=render_mode,
                log_level="warning",
                fig=fig,
                store_state_history=store_state_history,
            )

        @property
        def _define_actions(self):
            actions = [ah.IpAction(), ah.NbiAction(), ah.EcrhAction()]

            return actions

        @property
        def _define_observation(self):
            return oh.AllObservation()

        @property
        def _define_torax_config(self):
            return {
                "config": CONFIG,
                "discretization": "fixed",
                "ratio_a_sim": 1,
            }

        def _define_reward(self, state, next_state, action): 
            # Not used in this example
            return 0.0

.. code-block:: python

    class IterHybridAgent(BaseAgent):  # noqa: D101
        def __init__(self, action_space):  # noqa: D107
            super().__init__(action_space=action_space)
            self.time = 0

        def act(self, observation) -> dict:  # noqa: D102
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
state variables. As an illustration, the figure below shows current density profiles 
at selected times: 50s, 99s, 105s, and 150s.

.. grid:: 2

    .. grid-item::
        .. image:: Images/comparison_50.jpg

    .. grid-item::
        .. image:: Images/comparison_99.jpg

    .. grid-item::
        .. image:: Images/comparison_105.jpg

    .. grid-item::
        .. image:: Images/comparison_150.jpg
    
*Snapshots of current density at different times (native TORAX: dashed lines, 
Gym-TORAX: solid lines).*

This confirms that Gym-TORAX faithfully reproduces TORAX simulations and can be 
safely used as a control interface. Our complete code is available
`here <https://github.com/antoine-mouchamps/gymtorax/blob/main/examples/iter_hybrid.py>`_.