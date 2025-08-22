"""
TORAX Action Handler Module.

This module provides an abstract framework for defining and managing actions
in TORAX plasma simulations. Actions represent controllable parameters that
can be modified during the simulation to influence plasma behavior.

The Action class is designed to be extended by users to create custom actions
for specific control parameters. Each action has a unique name, dimensionality,
bounds, and knows how to map itself to TORAX configuration dictionaries and
which state variables it affects.

Classes:
    Action: Abstract base class for all action types (user-extensible)
    ActionHandler: Internal container and manager for multiple actions
    IpAction: Action for plasma current control
    VloopAction: Action for loop voltage control  
    EcrhAction: Action for electron cyclotron resonance heating
    NbiAction: Action for neutral beam injection

Example:
    Create a custom action by extending the Action class:
    
    >>> class CustomAction(Action):
    ...     name = "MyCustomAction"
    ...     dimension = 2
    ...     default_min = [0.0, -1.0]
    ...     default_max = [10.0, 1.0]
    ...     config_mapping = {
    ...         ('some_config', 'param1'): 0,
    ...         ('some_config', 'param2'): 1
    ...     }
    ...     state_var = (('scalars', 'param1'), ('scalars', 'param2'))
    
    Use an existing action:
    
    >>> ip_action = IpAction()
    >>> ip_action.set_values([1.5e6])  # 1.5 MA plasma current
"""

from abc import ABC
from typing import Any
from numpy.typing import NDArray
from gymnasium import spaces

import numpy as np
import logging

from torax._src.config.profile_conditions import _MIN_IP_AMPS


class Action(ABC):
    """
    Abstract base class for all TORAX simulation actions.
    
    An action represents a controllable parameter or set of parameters that can
    influence plasma behavior. Each action has bounds, current values, and knows
    how to map itself to TORAX configuration dictionaries.
    
    This class is designed to be extended by users to create custom actions for
    specific control parameters. Subclasses must define the class attributes
    to specify the action dimensionality, bounds, and configuration mapping.
    
    Class Attributes (must be overridden by subclasses):
        name (str): Unique identifier for this action type
        dimension (int): Number of parameters controlled by this action
        default_min (list[float]): Default minimum values for parameters
        default_max (list[float]): Default maximum values for parameters
        config_mapping (dict[tuple[str, ...], int]): Mapping from configuration
            paths to parameter indices. Keys are tuples representing the nested
            path in the config dictionary, values are parameter indices.
        state_var (tuple[tuple[str, ...], ...]): Tuple of tuples specifying the
            state variables directly modified by this action. Each inner tuple
            contains the path to a state variable (e.g., ('scalars', 'Ip') or 
            ('profiles', 'p_ecrh_e')).
        
    Instance Attributes:
        values (list[float]): Current parameter values
        dtype (np.dtype): NumPy data type for action arrays (default: np.float64)
    
    Example:
        Create a custom action for controlling two parameters:
        
        >>> class TwoParamAction(Action):
        ...     name = "CustomTwoParam"
        ...     dimension = 2
        ...     default_min = [0.0, -5.0]
        ...     default_max = [10.0, 5.0]
        ...     config_mapping = {
        ...         ('section', 'param1'): 0,
        ...         ('section', 'param2'): 1
        ...     }
        ...     state_var = (('scalars', 'param1'), ('scalars', 'param2'))
        >>> action = TwoParamAction()
    """
    
    # Class-level attributes to be overridden by subclasses
    dimension: int
    name: str
    default_min: list[float]
    default_max: list[float] 
    config_mapping: dict[tuple[str, ...], int]
    state_var: tuple[tuple[str]] = ()
    
    def __init__(
        self, 
        min: list[float] | None = None, 
        max: list[float] | None = None, 
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Initialize an Action instance.
        
        Args:
            min: Custom minimum bounds for each parameter. If None, uses the
                class default_min values. Must have length equal to dimension.
            max: Custom maximum bounds for each parameter. If None, uses the
                class default_max values. Must have length equal to dimension.
            dtype: NumPy data type for the action arrays (default: np.float64).
                Used for creating action spaces.
                
        Raises:
            ValueError: If any of the following conditions are met:
                - name class attribute is not defined
                - dimension class attribute is not defined or not a positive integer
                - config_mapping class attribute is not defined
                - default_min or default_max don't match the dimension
                - provided min or max don't match the dimension

        """
        # Validate that required class attributes are properly defined
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name' class attribute")
        
        if not isinstance(self.name, str):
            raise TypeError(f"{self.__class__.__name__} 'name' class attribute must be a string")

        if self.dimension is None:
            raise ValueError(f"{self.__class__.__name__} must define 'dimension' class attribute")
        
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self.dimension}")
        
        if self.config_mapping is None:
            raise ValueError(f"{self.__class__.__name__} must define 'config_mapping' class attribute")

        if not isinstance(self.default_min, list) or len(self.default_min) != self.dimension:
            raise ValueError(f"default_min must be a list of length {self.dimension}")

        if not isinstance(self.default_max, list) or len(self.default_max) != self.dimension:
            raise ValueError(f"default_max must be a list of length {self.dimension}")

        # Initialize minimum and maximum bounds for the action parameters
        # Use provided values or defaults
        self._min = min if min is not None else self.default_min
        self._max = max if max is not None else self.default_max
        
        # Validate bounds dimensions
        if self._min is not None and len(self._min) != self.dimension:
            raise ValueError(
                f"Invalid min bounds dimension: expected {self.dimension}, got {len(self._min)}"
            )
        if self._max is not None and len(self._max) != self.dimension:
            raise ValueError(
                f"Invalid max bounds dimension: expected {self.dimension}, got {len(self._max)}"
            )
        # Default value
        self.values = self._min
        
        self.dtype = dtype

    @property
    def min(self) -> list[float]:
        """
        Minimum bounds for this action parameters.
        
        Returns:
            list[float]: List of minimum values, one for each parameter
                controlled by this action.
        """
        return self._min

    @property
    def max(self) -> list[float]:
        """
        Maximum bounds for this action parameters.
        
        Returns:
            list[float]: List of maximum values, one for each parameter
                controlled by this action.
        """
        return self._max

    def set_values(self, values: list[float]) -> None:
        """
        Set the current parameter values for this action.

        Args:
            values (list[float]): The new parameter values to set.
        
        Raises:
            ValueError: If the length of the values list does not match the expected dimension.
        """
        if len(values) != self.dimension:
            raise ValueError(f"Expected {self.dimension} values, got {len(values)}")
        self.values = values

    def init_dict(self, config_dict: dict[str, Any]) -> None:
        """
        Initialize a TORAX configuration dictionary with this action parameters.
        
        This method sets up the configuration dictionary with the action current
        values at time=0, creating the proper time-dependent parameter structure
        expected by TORAX.
        
        Args:
            config_dict: The TORAX configuration dictionary to initialize.
                Must have the nested structure expected by this action
                config_mapping.
                
        Raises:
            KeyError: If the configuration dictionary doesn't have the expected
                structure for this action parameters.
        """
        try:
            self._apply_mapping(config_dict, time=0, warning=True)
        except Exception as e:
            raise KeyError(
                f"An error occurred while initializing the action in the dictionary: key {e} is missing"
            )

    def update_to_config(self, config_dict: dict[str, Any], time: float) -> None:
        """
        Update a TORAX configuration dictionary with new action values.
        
        This method updates the time-dependent parameters in the configuration
        dictionary with the action current values at the specified time.
        
        Args:
            config_dict: The TORAX configuration dictionary to update.
                Must have been previously initialized with init_dict.
            time: Simulation time for this update. Must be > 0.
        
        Note:
            The configuration dictionary must have been initialized with
            init_dict before calling this method.
        """
        self._apply_mapping(config_dict, time=time, warning = False)

    def _apply_mapping(self, config_dict: dict[str, Any], time: float, warning: bool) -> None:
        """
        Apply the action values to a TORAX configuration dictionary.
        
        This method traverses the configuration dictionary using the paths defined
        in config_mapping and sets the appropriate values. For time=0 (initialization),
        it creates new time-dependent parameter entries. For time>0, it updates
        existing entries.
        
        Args:
            config_dict: The TORAX configuration dictionary to modify
            time: Simulation time. If 0, initializes new time-dependent parameters.
                If >0, updates existing time-dependent parameters.
            warning: If True, emits a warning when overwriting existing values.

        Note:
            This is an internal method used by init_dict and update_to_config.
            The configuration format follows TORAX conventions where time-dependent
            parameters are stored as ({time: value, ...}, "STEP") tuples.
        """
        for dict_path, idx in self.config_mapping.items():
            # drill down into config_dict
            d = config_dict
            
            for key in dict_path[:-1]:
                d = d[key]

            key = dict_path[-1]
            if time == 0:
                #Check there is no value associated to the existing key
                if d[key] != {} and warning:
                    logging.warning(f"WARNING: overwriting existing value for key: {key}")
                d[key] = ({0: self.values[idx]}, "STEP")
            else:
                d[key][0].update({time: self.values[idx]})
                
    def get_mapping(self) -> dict[tuple[str, ...], int]:
        """
        Get the mapping of configuration dictionary paths to action parameter indices.

        Returns:
            dict[tuple[str, ...], int]: Mapping of config dictionary paths to action parameter indices.
        """
        return self.config_mapping

    def get_state_variables(self) -> tuple[tuple[str]]:
        """
        Get the state variables modified by the action.

        Returns:
            tuple[tuple[str]]: A tuple of tuples, each containing the state variable
            names modified by the action.
        """
        return self.state_var

    def __repr__(self) -> str:
        """
        Return a string representation of the action.
        
        Returns:
            str: String showing the action class name, current values, and bounds.
        """
        return f"{self.__class__.__name__}(values={self.values}, min={self.min}, max={self.max})"


class ActionHandler:
    """
    Internal container and manager for multiple actions.
    
    This class is used internally by the gymtorax framework to manage collections
    of actions.
    
    Args:
        actions: List of Action instances to manage.
        
    Attributes:
        actions: Internal list of managed actions.
    """
    
    def __init__(self, actions: list[Action]) -> None:
        """
        Initialize the ActionHandler with a list of actions.
        
        Args:
            actions: List of Action instances to manage.
        """
        self._actions = actions
        self._validate_action_handler()
        

    def get_actions(self) -> list[Action]:
        """
        Get the list of managed actions.
        
        Returns:
            list[Action]: List of Action instances managed by this handler.            
        """
        return self._actions


    def update_actions(self, action_array: NDArray) -> None:
        """
        Update the current values of all managed actions.
        
        Args:
            action_array: Array of new values for each action values.
                Must match the total number of parameters across all actions.
                
        Raises:
            ValueError: If the length of action_array does not match the
                total number of parameters in all managed actions.
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
    
    
    def build_action_space(self) -> spaces.Dict:
        """
        Build a Gymnasium Dict action space from all managed actions.
        
        Creates a dictionary-based action space where each key corresponds to 
        an action's name and each value is a Box space with the action's bounds
        and data type.
        
        Returns:
            spaces.Dict: Dictionary action space with action names as keys and
                Box spaces as values. Each Box space uses the action's min/max
                bounds and dtype for proper numerical handling.
        """
        
        return spaces.Dict({
            action.name: spaces.Box(low=np.array(action.min),
                                    high=np.array(action.max),
                                    dtype=action.dtype)
            for action in self.get_actions()
        })


    def _validate_action_handler(self) -> None:
        """
        Validates the action handler to ensure action parameters are unique
        and mutually exclusive.

        This function performs two main checks:
        1. It verifies that no duplicate parameters exist across all actions.
        2. It ensures that 'Ip' and 'Vloop' actions are not present simultaneously,
        as TORAX can only use one or the other.

        Raises:
            ValueError: If duplicate parameters are found or if both 'Ip'
                        and 'Vloop' actions are present.
        """
        seen_keys = set()
        seen_names = set()
        for action in self._actions:
            for key in action.get_mapping().keys():
                if key in seen_keys:
                    raise ValueError(f"Duplicate action parameter detected: {key}")
                seen_keys.add(key)
            if action.name in seen_names:
                raise ValueError(f"Duplicate action name detected: {action.name}")
            seen_names.add(action.name)

        # Check for exclusive presence of Ip or Vloop actions (through their keys)
        if ('profile_conditions', 'v_loop_lcfs') in seen_keys and \
           ('profile_conditions', 'Ip') in seen_keys:
            raise ValueError(
                "Cannot have both Ip and Vloop actions at the same time."
            )

# =============================================================================
# Pre-configured Action Examples
# =============================================================================
# The following classes are example implementations of actions used
# in TORAX plasma simulations. Users can use these directly.


class IpAction(Action):
    """
    Example action for controlling plasma current (Ip).
    
    This action controls the plasma current parameter in TORAX simulations.
    It is a single-parameter action with non-negative bounds.
    
    Class Attributes:
        name: "Ip"
        dimension: 1 (single parameter)
        default_min: [_MIN_IP_AMPS] (minimum current per TORAX requirements)
        default_max: [np.inf]
        config_mapping: Maps to ('profile_conditions', 'Ip')
        state_var: {'scalars': ['Ip']} - directly modifies plasma current scalar
        
    Example:
        >>> ip_action = IpAction()
        >>> ip_action.set_values([1.5e6])  # 1.5 MA plasma current
    """
    name = "Ip"
    dimension = 1
    default_min = [_MIN_IP_AMPS] # TORAX requirements
    default_max = [np.inf]
    config_mapping = {('profile_conditions', 'Ip'): 0}
    state_var = {'scalars': ['Ip']}


class VloopAction(Action):
    """
    Example action for controlling loop voltage at the last closed flux surface.
    
    This action controls the loop voltage parameter (v_loop_lcfs) in TORAX
    simulations. It is a single-parameter action with non-negative bounds.
    
    Class Attributes:
        name: "V_loop"
        dimension: 1 (single parameter)
        default_min: [0.0]
        default_max: [np.inf]
        config_mapping: Maps to ('profile_conditions', 'v_loop_lcfs')
        state_var: {'scalars': ['v_loop_lcfs']} - directly modifies loop voltage scalar
        
    Example:
        >>> vloop_action = VloopAction()
        >>> vloop_action.set_values([2.5])  # 2.5 V loop voltage
    """
    name = "V_loop"
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    config_mapping = {('profile_conditions', 'v_loop_lcfs'): 0}
    state_var = {'scalars': ['v_loop_lcfs']}


class EcrhAction(Action):
    """
    Example action for controlling Electron Cyclotron Resonance Heating (ECRH).
    
    This action controls three ECRH parameters: total power, Gaussian location,
    and Gaussian width of the heating profile.
    
    Class Attributes:
        name: "ECRH"
        dimension: 3 (power, location, width)
        default_min: [0.0, 0.0, 0.0]
        default_max: [np.inf, np.inf, np.inf]
        config_mapping: Maps to ECRH source parameters
        state_var: {'scalars': ['P_ecrh_e']} -
                   modifies total electron-cyclotron power scalar
        
    Parameters:
        - Index 0: Total power (P_total) in Watts
        - Index 1: Gaussian location (gaussian_location) - normalized radius [0,1]
        - Index 2: Gaussian width (gaussian_width) - profile width parameter
    
    Example:
        >>> ecrh_action = EcrhAction()
        >>> ecrh_action.set_values([5e6, 0.3, 0.1])  # 5MW, r/a=0.3, width=0.1
    """
    name = "ECRH"
    dimension = 3  # power, location, width
    default_min = [0.0, 0.0, 0.01]
    default_max = [np.inf, 1.0, np.inf]  
    config_mapping = {
        ('sources', 'ecrh', 'P_total'): 0,
        ('sources', 'ecrh', 'gaussian_location'): 1,
        ('sources', 'ecrh', 'gaussian_width'): 2
    }
    state_var = {'scalars': ['P_ecrh_e']}


class NbiAction(Action):
    """
    Example action for controlling Neutral Beam Injection (NBI).
    
    This action controls four NBI parameters: heating power, current drive power,
    Gaussian location, and Gaussian width. Both heating and current drive
    components share the same spatial profile (location and width).
    
    Class Attributes:
        name: "NBI"
        dimension: 4 (heating power, current power, location, width)
        default_min: [0.0, 0.0, 0.0, 0.01]
        default_max: [np.inf, np.inf, 1.0, np.inf]
        config_mapping: Maps to generic heat and current source parameters in TORAX configuration
        state_var: {'scalars': ['P_aux_generic_total', 'I_aux_generic']} -
                   modifies total auxiliary power and current scalars

    Parameters:
        - Index 0: Heating power (generic_heat P_total) in Watts
        - Index 1: Current drive power (generic_current I_generic) in Amperes
        - Index 2: Gaussian location (shared by heat and current) - normalized radius [0,1]
        - Index 3: Gaussian width (shared by heat and current) - profile width parameter
    
    Example:
        >>> nbi_action = NbiAction()
        >>> nbi_action.set_values([10e6, 2e6, 0.4, 0.2])
        >>> # 10MW heating, 2MA current drive, r/a=0.4, width=0.2
    """
    name = "NBI"
    dimension = 4  # heating power, current power, location, width
    default_min = [0.0, 0.0, 0.0, 0.01]
    default_max = [np.inf, np.inf, 1.0, np.inf]
    config_mapping = {
        ('sources', 'generic_heat', 'P_total'): 0,
        ('sources', 'generic_current', 'I_generic'): 1,
        ('sources', 'generic_heat', 'gaussian_location'): 2,
        ('sources', 'generic_heat', 'gaussian_width'): 3,
        ('sources', 'generic_current', 'gaussian_location'): 2,  # Shared location
        ('sources', 'generic_current', 'gaussian_width'): 3,     # Shared width
    }
    state_var = {'scalars': ['P_aux_generic_total', 'I_aux_generic']}
