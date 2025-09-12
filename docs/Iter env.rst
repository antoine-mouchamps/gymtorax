Environment description
=======================

The environment ``IterHybridEnv`` is derived from the ITER hybrid ramp-up scenario 
provided with TORAX tutorial and adapted from [Citrin_2010]. In this scenario, the plasma 
first undergoes a ramp-up phase (0–100 s) in L-mode (low confinement), followed by 
a nominal phase (100–150 s) in H-mode (high confinement). Both the control and 
simulation time steps are set to 1 second.

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

        def _define_action_space(self):
            actions = [ah.IpAction(), ah.NbiAction(), ah.EcrhAction()]

            return actions

        def _define_observation_space(self):
            return oh.AllObservation()

        def _get_torax_config(self):
            return {
                "config": CONFIG, # -> see Configuration Example
                "discretization": "fixed",
                "ratio_a_sim": 1,
            }

        def _compute_reward(self, state, next_state, action): 
            # Customize weights and sigma as needed
            weight_list = [2, 1, 2, 1, 1, 1]

            def _is_H_mode():
                if (
                    next_state["profiles"]["T_e"][0] > 10
                    and next_state["profiles"]["T_i"][0] > 10
                ):
                    return True
                else:
                    return False

            def _r_fusion_gain():
                fusion_gain = rw.get_fusion_gain(next_state) / 10  # Normalize to [0, 1]
                if _is_H_mode():
                    return fusion_gain
                else:
                    return 0

            def _r_h98():
                h98 = rw.get_h98(next_state)
                if _is_H_mode():
                    if h98 >= 1:
                        return 1
                    else:
                        return 0
                else:
                    return 0

            def _r_q_min():
                q_min = rw.get_q_min(next_state)
                if q_min <= 1:
                    return 0
                elif q_min > 1:
                    return 1

            def _r_q_95():
                q_95 = rw.get_q95(next_state)
                if q_95 <= 3:
                    return 0
                else:
                    return 1

            # Calculate individual reward components
            r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
            r_h98 = weight_list[2] * _r_h98() / 50
            r_q_min = weight_list[3] * _r_q_min() / 150
            r_q_95 = weight_list[4] * _r_q_95() / 150

            total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_95
            return total_reward


