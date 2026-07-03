"""Test Gymnasium environment compliance using env_checker."""

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gymtorax  # noqa: F401
from tests.conftest import ALL_ENV_IDS


class TestEnvironmentCompliance:
    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_env_checker_compliance(self, make_env, env_id):
        env = make_env(env_id)
        check_env(env.unwrapped, skip_render_check=True)

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_step_return_types(self, make_env, env_id):
        env = make_env(env_id)
        env.reset()

        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(obs, dict)
            assert isinstance(reward, int | float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            if terminated or truncated:
                break

    def test_iter_hybrid_obs_structure(self, make_env):
        env = make_env("gymtorax/IterHybrid-v0")
        obs, _ = env.reset()

        assert isinstance(obs, dict)
        assert "profiles" in obs
        assert "scalars" in obs
        assert isinstance(env.action_space, gym.spaces.Dict)

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_observation_space_bounds(self, make_env, env_id):
        env = make_env(env_id)
        obs, _ = env.reset()

        assert env.observation_space.contains(obs), (
            "Initial observation is not within observation space bounds"
        )

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs), (
                "Step observation is not within observation space bounds"
            )

            if terminated or truncated:
                break

    def test_environment_reset_consistency(self, make_env):
        env = make_env("gymtorax/IterHybrid-v0")

        for i in range(5):
            obs, info = env.reset(seed=i)

            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            assert env.observation_space.contains(obs)

            action = env.action_space.sample()
            step_obs, reward, terminated, truncated, step_info = env.step(action)

            assert isinstance(step_obs, dict)
            assert env.observation_space.contains(step_obs)
