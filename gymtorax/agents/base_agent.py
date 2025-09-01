"""Base Agent Module for TORAX Gymnasium Environments.

This module provides the abstract base class for creating agents that can interact
with TORAX plasma simulation environments. The BaseAgent class defines the common
interface that all agents must implement to work with the gymtorax framework.

The agent architecture follows the standard reinforcement learning pattern where
agents observe the environment state and produce actions to control plasma parameters.
Concrete agent implementations can include rule-based controllers, PID controllers,
or reinforcement learning-based agents.

Classes:
    BaseAgent: Abstract base class defining the agent interface

Example:
    Create a custom agent by extending BaseAgent:

    >>> class RandomAgent(BaseAgent):
    ...     def __init__(self, action_space):
    ...         super().__init__(action_space)
    ...
    ...     def act(self, observation):
    ...         # Generate random actions within the action space bounds
    ...         return self.action_space.sample()
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for agents interacting with TORAX environments.

    This class defines the standard interface that all agents must implement to
    work with TORAX plasma simulation environments. It provides the foundation
    for creating various types of controllers including rule-based systems,
    classical control algorithms, and reinforcement learning agents.

    The agent receives observations from the environment containing plasma state
    information and must produce actions within the defined action space to
    control plasma parameters like heating power, current drive, or gas injection.

    Attributes:
        action_space (gym.Space): The action space defining valid actions
            that can be sent to the environment.

    Abstract Methods:
        act: Generate an action based on the current observation.

    Example:
        >>> class RandomAgent(BaseAgent):
        ...     def act(self, observation):
        ...         return self.action_space.sample()
    """

    def __init__(self, action_space):
        """Initialize the base agent with an action space.

        Args:
            action_space (gym.Space): The action space from the TORAX environment
                defining the valid range and structure of actions. This typically
                includes bounds for plasma control parameters like heating power,
                current drive, gas injection rates, etc.
        """
        self.action_space = action_space

    @abstractmethod
    def act(self, observation):
        """Generate an action based on the current plasma state observation.

        This method must be implemented by all concrete agent subclasses to define
        how the agent responds to plasma state information. The observation contains
        current plasma parameters, and the agent must return a valid action within
        the defined action space.

        Args:
            observation (dict): Current plasma state observation from the environment.

        Returns:
            action (dict): Action to be taken, structured according to the
                action space specification.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
