class RandomAgent:  # noqa: D101
    """Agent for the ITER hybrid scenario.

    This agent produces random actions within the action space.
    """

    def __init__(self, action_space):
        """Initialize the agent with the given action space."""
        self.action_space = action_space

    def act(self, observation) -> dict:
        """Compute the next action based on the current observation and internal time.

        Returns:
            dict: Action dictionary for the environment.
        """
        action = self.action_space.sample()

        return action
