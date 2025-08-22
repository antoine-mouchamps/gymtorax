

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    @abstractmethod
    def act(self, observation):
        raise NotImplementedError