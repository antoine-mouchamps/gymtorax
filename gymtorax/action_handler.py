from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Action(ABC):
    # Class-level attributes to be overridden by subclasses
    dimension: int = None
    default_min: list = None  
    default_max: list = None
    map: dict = None #Will store the path in the config file for the action
    
    def __init__(self, min_vals: list = None, max_vals: list = None, values: list = None):
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
        
        if values and len(values) != self.dimension:
            raise ValueError(f"Invalid dimension {len(self.values)}, must be {self.dimension}")
        self.values = values or [0.0] * self.dimension
        

    @property
    def min(self) -> list:
        """Minimum bounds for this action."""
        return self._min
    
    @property
    def max(self) -> list:
        """Maximum bounds for this action."""
        return self._max
    
    def set_val(self, values: float | list[float]) -> None:
        """Update values stored in this action"""
        if len(values) != self.dimension:
            raise ValueError(f"The length of the list is not appropriate. '{self.dimension}' was expected.") 
        self.values = values
    
    #Only used in action classes
    def _apply_mapping(self, config_dict, time: float) -> None:
        """
        list_dict is the list of values stored by an action
        mode = 'init'   -> assign tuple (value, 'STEP')
        mode = 'update' -> call update() on the first element
        """
        for dict_path, idx in self.map.items():
            # drill down into config_dict
            d = config_dict
            for key in dict_path[:-1]:
                d = d[key]

            key = dict_path[-1]
            if time == 0:
                d[key] = ({0: self.values[idx]}, "STEP")
            else:
                d[key][0].update({time: self.values[idx]})
    
    def init_dict(self, config_dict: dict) -> None:
        """Verify if config_dict is convenient for this action"""
        try:
            self._apply_mapping(config_dict, time = 0)
        except Exception as e:
            raise KeyError(
                f"An error occurred while initializing the action in the dictionary: {e}"
            )
    
    def update_to_config(self, config_dict: dict, time: float) -> None:
        """Update the config_dict with the values stored in this action"""
        self._apply_mapping(config_dict, time = time)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class ActionHandler:
    def __init__(self, actions: list[Action]):
        self.actions = actions
    def get_actions(self) -> list[Action]:
        """
        Get the list of managed actions.
        """
        return self._actions


    def update_actions(self, action_array: NDArray) -> None:
        """
        Update the current values of all managed actions.
        """
        total_params = sum(action.dimension for action in self._actions)
        if len(action_array) != total_params:
            raise ValueError(
                f"Expected {total_params} action parameters, got {len(action_array)}"
            )
        
        idx = 0
        for action in self._actions:
            action.set_values(action_array[idx:idx + action.dimension])
            idx += action.dimension


    def get(self) -> list[Action]:
        return self.actions

class IpAction(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    map = {('profile_conditions', 'Ip'): 0}

class VloopAction(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    map = {('profile_conditions', 'v_loop_lcfs'): 0}

class EcrhAction(Action):
    dimension = 3 # power, loc, width
    default_min = [0.0, 0.0, 0.0]
    default_max = [np.inf, np.inf, np.inf]  
    map = {
        ('sources','ecrh','P_total'): 0,
        ('sources','ecrh','gaussian_location'): 1,
        ('sources','ecrh','gaussian_width'): 2
    }
    
class NbiAction(Action):
    dimension = 4 # power heating, power current, loc, width
    default_min = [0.0, 0.0, 0.0, 0.0]
    default_max = [np.inf, np.inf, np.inf, np.inf]
    map = {
        ('sources','generic_heat','P_total'): 0,
        ('sources','generic_current','I_generic'): 1,
        ('sources','generic_heat','gaussian_location'): 2,
        ('sources','generic_heat','gaussian_width'): 3,
        ('sources','generic_current','gaussian_location'): 2,
        ('sources','generic_current','gaussian_width'): 3,
    }
    