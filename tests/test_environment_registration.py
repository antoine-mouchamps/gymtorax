"""Test environment registration with Gymnasium."""

import gymnasium as gym
import pytest

# Import gymtorax to trigger environment registration
import gymtorax  # noqa: F401


class TestEnvironmentRegistration:
    """Test that environments are properly registered with Gymnasium."""

    def test_iter_hybrid_env_registration(self):
        """Test that IterHybridEnv is registered and can be created with gym.make()."""
        # Test that the environment can be created
        env = gym.make("gymtorax/IterHybrid-v0")

        # Basic checks
        assert env is not None
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Test that it's the correct type
        from gymtorax.envs.iter_hybrid_env import IterHybridEnv

        assert isinstance(env.unwrapped, IterHybridEnv)

        env.close()

    def test_test_env_registration(self):
        """Test that TestEnv is registered and can be created with gym.make()."""
        # Test that the environment can be created
        env = gym.make("gymtorax/Test-v0")

        # Basic checks
        assert env is not None
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Test that it's the correct type
        from examples.test_env import TestEnv

        assert isinstance(env.unwrapped, TestEnv)

        env.close()

    def test_environment_creation_with_kwargs(self):
        """Test that environments can be created with custom parameters."""
        # Test IterHybridEnv with custom parameters
        env1 = gym.make(
            "gymtorax/IterHybrid-v0",
            render_mode="rgb_array",
            log_level="info",
            store_state_history=True,
        )
        assert env1 is not None
        env1.close()

        # Test TestEnv with custom parameters
        env2 = gym.make("gymtorax/Test-v0", render_mode="human", log_level="debug")
        assert env2 is not None
        env2.close()

    def test_registered_environments_list(self):
        """Test that our environments appear in the registry."""
        import gymnasium.envs.registration as registration

        env_ids = list(registration.registry.keys())

        assert "gymtorax/IterHybrid-v0" in env_ids
        assert "gymtorax/Test-v0" in env_ids

    @pytest.mark.parametrize("env_id", ["gymtorax/IterHybrid-v0", "gymtorax/Test-v0"])
    def test_environment_basic_interface(self, env_id):
        """Test basic Gymnasium interface compliance for all registered environments."""
        env = gym.make(env_id)

        try:
            # Test reset
            observation, info = env.reset()
            assert isinstance(observation, dict)
            assert isinstance(info, dict)

            # Test that action space and observation space are defined
            assert env.action_space is not None
            assert env.observation_space is not None

            # Test that we can sample a random action
            action = env.action_space.sample()
            assert action is not None

        finally:
            env.close()
