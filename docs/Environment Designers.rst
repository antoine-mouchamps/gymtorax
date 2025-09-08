Environment Designers
===========================

The **Gym-TORAX** package provides a powerful framework for plasma physics specialists 
to explore and create new environments. It is designed as a flexible toolkit that allows 
users with a background in plasma physics to develop and test their own scenarios, 
configurations, and experiments. This includes defining new actions, observations, 
the TORAX configuration (for the plasma properties) and rewards.


Creation of New Environments
------------------------------

Base Environment
^^^^^^^^^^^^^^^^^

.. autoclass:: gymtorax.envs.base_env.BaseEnv
    :no-index:

Abstract Methods and Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: gymtorax.envs.base_env.BaseEnv._define_actions
    :no-index:

.. autoattribute:: gymtorax.envs.base_env.BaseEnv._define_observation
    :no-index:

.. autoattribute:: gymtorax.envs.base_env.BaseEnv._define_torax_config
    :no-index:

.. automethod:: gymtorax.envs.base_env.BaseEnv._define_reward
    :no-index:

Here is a simple example of how to create a new environment by extending the base class:

.. code-block:: python

    from gymtorax.envs.base_env import BaseEnv
    import gymtorax.action_handler as ah 
    import gymtorax.observation_handler as oh
    import gymtorax.rewards as rw

    class CustomEnv(BaseEnv):
        def _define_actions(self):
            actions = [ah.IpAction(),]
            return actions
            
        def _define_observation(self):
            return oh.AllObservation()

        def _define_torax_config(self):
            return {"config": CONFIG, 
                "discretization": "auto", 
                "delta_t_a": 1.0}

        def _define_reward(self, current_state, next_state, action):
            Q = rw.get_fusion_gain(next_state)
            q_min = rw.get_q_min(next_state)
            w_Q, w_qmin = 1.0, 1.0

            def q_min_function():
            if q_min <= 1:
                return 0
            elif q_min > 1:
                return 1

            return w_Q * Q + w_qmin * q_min_function()


Creation of New Actions
------------------------------

.. autoclass:: gymtorax.action_handler.Action
    :no-index:

Creation of New Observations
------------------------------

.. autoclass:: gymtorax.observation_handler.Observation
    :no-index:

Creation of a configuration
------------------------------
The configuration needed in the method `_define_torax_config` is a dictionary which
is exactly the same as the one used in TORAX. You can find more details about the configuration
in the `TORAX documentation <https://torax.readthedocs.io/en/stable/configuration.html>`_.