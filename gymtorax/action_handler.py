from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Action(ABC):
    # Class-level attributes to be overridden by subclasses
    dimension: int = None
    default_min: list = None  
    default_max: list = None
    
    def __init__(self, min_vals: list = None, max_vals: list = None):
        # Validate dimension is defined
        if self.dimension is None:
            raise ValueError(f"{self.__class__.__name__} must define dimension class attribute")
        
        # Use provided values or defaults
        self._min = min_vals or self.default_min
        self._max = max_vals or self.default_max
        
        # Validate bounds dimensions
        if self._min is not None and len(self._min) != self.dimension:
            raise ValueError(f"Invalid min dimension {len(self._min)}, must be {self.dimension}")
        if self._max is not None and len(self._max) != self.dimension:
            raise ValueError(f"Invalid max dimension {len(self._max)}, must be {self.dimension}")
        
        # Set defaults if None
        if self._min is None:
            self._min = [0.0] * self.dimension
        if self._max is None:
            self._max = [np.inf] * self.dimension

    @property
    def min(self) -> list:
        """Minimum bounds for this action."""
        return self._min
    
    @property
    def max(self) -> list:
        """Maximum bounds for this action."""
        return self._max


    @abstractmethod
    def get_dict(self, time: float) -> dict:
        """Return the action as a dict for the simulator config."""
        pass
    
    def set_val(self, values: float | list[float]) -> None:
        """Update values stored in this action"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class ActionHandler:
    def __init__(self, actions: list[Action]):
        self.actions = actions

    def get(self):
        return self.actions

class IpAction(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    
    def __init__(self, min_vals: list = None, max_vals: list = None, value: float = None):
        super().__init__(min_vals, max_vals)
        self.value = value or 0.0

    def get_dict(self, time: float) -> dict:
        return {time: self.value}
    
    def set_val(self, value: float) -> None:
        self.value = value


class VloopAction(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    
    def __init__(self, min_vals: list = None, max_vals: list = None, value: float = None):
        super().__init__(min_vals, max_vals)
        self.value = value or 0.0

    def get_dict(self, time: float) -> dict:
        return {time: self.value}

    def set_val(self, value: float) -> None:
        self.value = value


class EcrhAction(Action):
    dimension = 3 # power, loc, width
    default_min = [0.0, 0.0, 0.0]
    default_max = [np.inf, np.inf, np.inf]  

    def __init__(self, min_vals: list = None, max_vals: list = None, values: list = None):
        super().__init__(min_vals, max_vals)
        self.values = values or [0.0] * self.dimension

    def get_dict(self, time: float) -> list[dict]:
        return [{time: self.values[0]}, {time: self.values[1]}, {time: self.values[2]}]
    
    def set_val(self, values: list[float]) -> None:
        if len(values) != self.dimension:
            raise ValueError(f"The length of the list is not appropriate. '{self.dimension}' was expected.")
        self.values = values


class NbiAction(Action):
    dimension = 4 # power heating, power current, loc, width
    default_min = [0.0, 0.0, 0.0, 0.0]
    default_max = [np.inf, np.inf, np.inf, np.inf]

    def __init__(self, min_vals: list = None, max_vals: list = None, values: list = None):
        super().__init__(min_vals, max_vals)
        self.values = values or [0.0] * self.dimension

    def get_dict(self, time: float) -> list[dict]:
        return [{time: self.values[0]},{time: self.values[1]},{time: self.values[2]},{time: self.values[3]}]

    def set_val(self, values: list[float]) -> None:
        if len(values) != self.dimension:
            raise ValueError(f"The length of the list is not appropriate. '{self.dimension}' was expected.")
        self.values = values
