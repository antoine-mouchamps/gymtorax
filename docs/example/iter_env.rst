Environment Description
=======================


The environment used for this example is the ITER hybrid ramp-up scenario, provided as an example configuration file in TORAX. This scenario consists of a power ramp-up phase of 100 seconds, followed by a nominal phase lasting 50 seconds. The first phase of 100 seconds takes place in so-called *L-mode* (low-confinement regime), while the nominal phase occurs in *H-mode* (high-confinement regime). These are two distinct plasma confinement regimes with different physical properties.

The environment is named ``IterHybridEnv``.

Actions
-------

The environment features three actions: 

- ``IpAction`` — total current (:math:`I_p`),
- ``NbiAction`` — neutral beam injection 
  (:math:`NBI`),
- ``EcrhAction`` — electron cyclotron 
  resonance heating (:math:`ECRH`).

Its action space :math:`\mathcal{A}` is bounded, and a ramp-rate limit is imposed on the total current.

Observations
------------

By default, the environment is fully observable and uses the ``AllObservation`` class 
with custom bounds applied to some variables.

Reward
------

The reward function is a linear combination of four elements:

.. math::
   \begin{equation}
   r = \alpha_Q\cdot g_Q + \alpha_{q_{min}}\cdot g_{q_{min}} + \alpha_{q_{95}}\cdot g_{q_{95}} + \alpha_\mathrm{H98}\cdot g_\mathrm{H98}\quad.
   \end{equation}

In this equation, :math:`\alpha_i` and :math:`g_i`, with :math:`i\in\{Q,q_{min},q_{95},\mathrm{H98}\}`, represent weights and functions, respectively. These are related to the fusion gain :math:`Q`, the minimum :math:`q_{min}` and edge :math:`q_{95}` safety factors, and the H-mode confinement quality factor :math:`\mathrm{H}98`, respectively.

This environment is used both for physical validation and for control experiments.

Here is the environment:

.. code-block:: python

    import gymtorax.action_handler as ah
    import gymtorax.observation_handler as oh
    from gymtorax.envs.base_env import BaseEnv

    CONFIG = {...}


    class IterHybridEnv(BaseEnv):

        def __init__(self, render_mode=None, **kwargs):

            # Set environment-specific defaults
            kwargs.setdefault("log_level", "warning")
            kwargs.setdefault("plot_config", "default")

            super().__init__(render_mode=render_mode, **kwargs)

        def _define_action_space(self):
            actions = [
                IpAction(
                    max=[15e6],  # 15 MA max plasma current
                    ramp_rate=[0.2e6],
                ),  # 0.2 MA/s ramp rate limit
                NbiAction(
                    max=[33e6, 1.0, 1.0],  # 33 MW max NBI power
                ),
                EcrhAction(
                    max=[20e6, 1.0, 1.0],  # 20 MW max ECRH power
                ),
            ]

            return actions

        def _define_observation_space(self):
            return AllObservation(custom_bounds_file="gymtorax/envs/iter_hybrid.json")

        def _get_torax_config(self):
            return {
                "config": CONFIG,
                "discretization": "fixed",
                "ratio_a_sim": 1,
            }

        def _compute_reward(self, state, next_state, action):
            weight_list = [1, 1, 1, 1]

            def _is_H_mode():
                if (
                    next_state["profiles"]["T_e"][0] > 10
                    and next_state["profiles"]["T_i"][0] > 10
                ):
                    return True
                else:
                    return False

            def _r_fusion_gain():
                fusion_gain = (
                    reward.get_fusion_gain(next_state) / 10
                )  # Normalize with ITER target
                if _is_H_mode():
                    return fusion_gain
                else:
                    return 0

            def _r_h98():
                h98 = reward.get_h98(next_state)
                if _is_H_mode():
                    if h98 <= 1:
                        return h98
                    else:
                        return 1
                else:
                    return 0

            def _r_q_min():
                q_min = reward.get_q_min(next_state)
                if q_min <= 1:
                    return q_min
                elif q_min > 1:
                    return 1

            def _r_q_95():
                q_95 = reward.get_q95(next_state)
                if q_95 / 3 <= 1:
                    return q_95 / 3
                else:
                    return 1

            # Calculate individual reward components
            r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
            r_h98 = weight_list[1] * _r_h98() / 50
            r_q_min = weight_list[2] * _r_q_min() / 150
            r_q_95 = weight_list[3] * _r_q_95() / 150

            total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_95

            return total_reward
