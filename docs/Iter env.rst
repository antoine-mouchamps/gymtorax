Environment description
=======================

The custom environment, ``IterHybridEnv``, uses the same TORAX configuration file 
as the ITER hybrid reference scenario. Control and simulation steps are set to 1 s.

Actions
-------

The environment defines three controllable inputs:

- :py:class:`IpAction()<gymtorax.action_handler.IpAction>` — plasma current (:math:`I_p`),
- :py:class:`NbiAction()<gymtorax.action_handler.NbiAction>` — neutral beam injection 
  (:math:`NBI`),
- :py:class:`EcrhAction()<gymtorax.action_handler.EcrhAction>` — electron cyclotron 
  resonance heating (:math:`ECRH`).

Observations
------------

By default, the environment is fully observable, using the 
:py:class:`AllObservation<gymtorax.observation_handler.AllObservation>` class 
with custom bounds applied to some variables.

Reward
------

The reward function combines four key terms:

- fusion gain (:math:`Q`),
- minimum safety factor (:math:`q_{min}`),
- edge safety factor (:math:`q_{95}`),
- H-mode confinement quality factor (:math:`H_{98}`).

This environment is used both for physical validation and for control experiments.

Here is the environment:

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
            # WIP
            return 0.0

