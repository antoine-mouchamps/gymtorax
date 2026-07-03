"""Test environment registration with Gymnasium."""

import os

import gymnasium as gym
import gymnasium.envs.registration as registration
import pytest

# Set matplotlib to a headless backend for CI environments. This must run before
# importing gymtorax, which imports matplotlib.pyplot at load time.
if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
    import matplotlib

    matplotlib.use("Agg")

import gymtorax  # noqa: F401  (import registers the environments with Gymnasium)
from gymtorax.envs import IterHybridEnv, TestEnv
from tests.conftest import ALL_ENV_IDS

# Expected concrete (unwrapped) class for each registered environment id.
ENV_CLASSES = {
    "gymtorax/IterHybrid-v0": IterHybridEnv,
    "gymtorax/Test-v0": TestEnv,
}


class TestEnvironmentRegistration:
    """Test that environments are properly registered with Gymnasium."""

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_env_registration(self, env_id):
        """Test that each environment is registered and creatable via ``gym.make()``."""
        env = gym.make(env_id)

        # Basic checks
        assert env is not None
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Test that it's the correct type
        assert isinstance(env.unwrapped, ENV_CLASSES[env_id])

        env.close()

    def test_environment_creation_with_kwargs(self):
        """Test that environments can be created with custom parameters."""
        # Test IterHybridEnv with custom parameters
        env1 = gym.make(
            "gymtorax/IterHybrid-v0",
            render_mode="rgb_array",
            log_level="info",
            store_history=True,
        )
        assert env1 is not None
        env1.close()

        # Test TestEnv with custom parameters
        # Skip human rendering in CI environments where no display is available
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            # Use rgb_array mode in CI to avoid Qt dependency
            env2 = gym.make(
                "gymtorax/Test-v0", render_mode="rgb_array", log_level="debug"
            )
        else:
            env2 = gym.make("gymtorax/Test-v0", render_mode="human", log_level="debug")
        assert env2 is not None
        env2.close()

    def test_registered_environments_list(self):
        """Test that our environments appear in the registry."""
        env_ids = list(registration.registry.keys())

        for env_id in ALL_ENV_IDS:
            assert env_id in env_ids

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
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
