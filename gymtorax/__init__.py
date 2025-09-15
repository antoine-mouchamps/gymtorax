from gymnasium.envs.registration import register

from .agents import IterHybridAgent, PIDAgent, RandomAgent
from .envs import BaseEnv, IterHybridEnv

# Register environments with Gymnasium
register(
    id="gymtorax/IterHybrid-v0",
    entry_point="gymtorax.envs:IterHybridEnv",
    kwargs={},
)

# Register the basic test environment for examples
register(
    id="gymtorax/Test-v0",
    entry_point="examples.test_env:TestEnv",
    kwargs={},
)

__all__ = ["BaseEnv", "IterHybridEnv", "PIDAgent", "IterHybridAgent", "RandomAgent"]
