"""Test Gymnasium environment compliance using env_checker.

This module tests that all gymtorax environments are compliant with the
Gymnasium interface using the official env_checker utility.
"""

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

# Import gymtorax to trigger environment registration
import gymtorax  # noqa: F401


class TestEnvironmentCompliance:
    """Test that all environments comply with Gymnasium interface standards."""

    @pytest.mark.parametrize("env_id", ["gymtorax/IterHybrid-v0", "gymtorax/Test-v0"])
    def test_env_checker_compliance(self, env_id):
        """Test environment compliance using gymnasium.utils.env_checker.check_env.

        This test uses the official Gymnasium environment checker to validate that
        the environment implements the correct interface and follows best practices.

        Args:
            env_id: The registered environment ID to test.
        """
        # Create the environment
        env = gym.make(env_id)

        try:
            # Run the official Gymnasium environment checker
            # This will validate:
            # - Correct action/observation space definitions
            # - Proper reset() and step() method signatures and return values
            # - Observation/action space containment checks
            # - Proper info dict handling
            # - Seed functionality
            # - Render modes (if applicable)
            check_env(env.unwrapped, skip_render_check=True)

        finally:
            env.close()

    def test_iter_hybrid_env_specific_compliance(self):
        """Test IterHybridEnv specific compliance checks."""
        env = gym.make("gymtorax/IterHybrid-v0")

        try:
            # Run comprehensive checks
            check_env(env.unwrapped, skip_render_check=True)

            # Additional checks specific to IterHybridEnv
            obs, info = env.reset()

            # Check observation structure
            assert isinstance(obs, dict), "Observation should be a dictionary"
            assert "profiles" in obs, "Observation should contain 'profiles'"
            assert "scalars" in obs, "Observation should contain 'scalars'"

            # Check that action space is a Dict space
            assert isinstance(env.action_space, gym.spaces.Dict), (
                "Action space should be a Dict space"
            )

            # Check that we can take a few steps
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # Validate step return values
                assert isinstance(obs, dict)
                assert isinstance(reward, int | float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                if terminated or truncated:
                    break

        finally:
            env.close()

    def test_test_env_specific_compliance(self):
        """Test TestEnv specific compliance checks."""
        env = gym.make("gymtorax/Test-v0")

        try:
            # Run comprehensive checks
            check_env(env.unwrapped, skip_render_check=True)

            # Additional checks specific to TestEnv
            obs, info = env.reset()

            # Check that we can take a few steps
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # Validate step return values
                assert isinstance(obs, dict)
                assert isinstance(reward, int | float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                if terminated or truncated:
                    break

        finally:
            env.close()

    @pytest.mark.parametrize("env_id", ["gymtorax/IterHybrid-v0", "gymtorax/Test-v0"])
    def test_action_space_sampling_and_bounds(self, env_id):
        """Test that action space sampling and bounds checking work correctly."""
        env = gym.make(env_id)

        try:
            # Sample multiple actions to test bounds
            for _ in range(10):
                action = env.action_space.sample()

                # Check that sampled action is within bounds
                assert env.action_space.contains(action), (
                    f"Sampled action {action} is not within action space bounds"
                )

                # Test that we can step with sampled action
                env.reset()
                obs, reward, terminated, truncated, info = env.step(action)

                # Basic validation of step return
                assert isinstance(obs, dict)
                assert isinstance(reward, int | float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

        finally:
            env.close()

    @pytest.mark.parametrize("env_id", ["gymtorax/IterHybrid-v0", "gymtorax/Test-v0"])
    def test_observation_space_bounds(self, env_id):
        """Test that observations are within the declared observation space bounds."""
        env = gym.make(env_id)

        try:
            obs, _ = env.reset()

            # Check initial observation is within bounds
            assert env.observation_space.contains(obs), (
                "Initial observation is not within observation space bounds"
            )

            # Take a few steps and check observations remain within bounds
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert env.observation_space.contains(obs), (
                    "Step observation is not within observation space bounds"
                )

                if terminated or truncated:
                    break

        finally:
            env.close()

    def test_environment_reset_consistency(self):
        """Test that reset() consistently returns valid observations."""
        env = gym.make("gymtorax/IterHybrid-v0")

        try:
            # Reset multiple times and check consistency
            for i in range(5):
                obs, info = env.reset(seed=i)

                # Check observation structure is consistent
                assert isinstance(obs, dict)
                assert isinstance(info, dict)
                assert env.observation_space.contains(obs)

                # Take one step to ensure the environment is functional after reset
                action = env.action_space.sample()
                step_obs, reward, terminated, truncated, step_info = env.step(action)

                assert isinstance(step_obs, dict)
                assert env.observation_space.contains(step_obs)

        finally:
            env.close()
