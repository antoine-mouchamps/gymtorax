"""TORAX Action Handler Module.

This module provides an abstract framework for defining and managing actions
in TORAX plasma simulations. Actions represent controllable parameters that
can be modified during the simulation to influence plasma behavior.

The `Action` class is designed to be extended by users to create custom actions
for specific control parameters. Each action has a unique name, dimensionality,
bounds, ramp rate limits, and knows how to map itself to TORAX configuration
dictionaries and which state variables it affects.

Classes:
    - `Action`: Abstract base class for all action types (user-extensible)
    - `ActionHandler`: Internal container and manager for multiple actions
    - `IpAction`: Action for plasma current control
    - `VloopAction`: Action for loop voltage control
    - `EcrhAction`: Action for electron cyclotron resonance heating
    - `NbiAction`: Action for neutral beam injection

Example:
    Create a custom action by extending the `Action` class:

    >>> class CustomAction(Action):
    ...     name = "MyCustomAction"
    ...     dimension = 2
    ...     default_min = [0.0, -1.0]
    ...     default_max = [10.0, 1.0]
    ...     default_ramp_rate = [0.1, None]  # First param limited, second unlimited
    ...     config_mapping = {
    ...         ('some_config', 'param1'): (0, 1),    # (index, factor)
    ...         ('some_config', 'param2'): (1, 1)     # (index, factor)
    ...     }
    ...     state_var = {'scalars': ['param1', 'param2']}

    Use an existing action:

    >>> ip_action = IpAction()
    >>> ip_action._set_values([1.5e6])  # 1.5 MA plasma current
"""

import copy
import logging
from abc import ABC
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from torax._src.config.profile_conditions import _MIN_IP_AMPS

# Set up logger for this module
logger = logging.getLogger(__name__)


class Action(ABC):
    """Abstract base class for all TORAX simulation actions.

    An action represents a controllable parameter or set of parameters that can
    influence plasma behavior. Each action has bounds, current values, and knows
    how to map itself to TORAX configuration dictionaries.

    This class is designed to be extended by users to create custom actions for
    specific control parameters. Subclasses must define the class attributes
    to specify the action dimensionality, bounds, and configuration mapping.

    Class Attributes:
        name (str): Unique identifier for this action type
        dimension (int): Number of parameters controlled by this action
        default_min (list[float]): Default minimum values for parameters
        default_max (list[float]): Default maximum values for parameters
        config_mapping (dict[tuple[str, ...], tuple[int, float]]): Mapping from configuration
            paths to parameter indices and scaling factors. Keys are tuples representing the nested
            path in the config dictionary, values are tuples of ``(parameter_index, scaling_factor)``.
        state_var (tuple[tuple[str, ...], ...]): Tuple of tuples specifying the
            state variables directly modified by this action. Each inner tuple
            contains the path to a state variable (e.g., ``('scalars', 'Ip')`` or
            ``('profiles', 'p_ecrh_e')``).

    Attributes:
        values (list[float]): Current parameter values
        ramp_rate (numpy.ndarray): Ramp rate limits for each parameter.
            ``numpy.inf`` indicates no ramp rate limit for that parameter.
        dtype (numpy.dtype): NumPy data type for action arrays (default: ``np.float64``)

    Example:
        Create a custom action for controlling two parameters:

        >>> class TwoParamAction(Action):
        ...     name = "CustomTwoParam"
        ...     dimension = 2
        ...     default_min = [0.0, -5.0]
        ...     default_max = [10.0, 5.0]
        ...     default_ramp_rate = [0.5, None]  # First param limited, second unlimited
        ...     config_mapping = {
        ...         ('section', 'param1'): (0, 1),      # No scaling
        ...         ('section', 'param2'): (1, 0.5)     # Scale by 0.5
        ...     }
        ...     state_var = {'scalars': ['param1', 'param2']}
        >>> action = TwoParamAction()
    """

    # Class-level attributes to be overridden by subclasses
    dimension: int
    name: str
    default_min: list[float]
    default_max: list[float]
    default_ramp_rate: list[float | None]
    config_mapping: dict[tuple[str, ...], int]
    state_var: dict[str, list[str]] = {}

    def __init__(
        self,
        min: list[float] | None = None,
        max: list[float] | None = None,
        ramp_rate: list[float | None] | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Initialize an Action instance.

        Args:
            min: Custom minimum bounds for each parameter. If ``None``, uses the
                class ``default_min`` values. Must have length equal to ``dimension``.
            max: Custom maximum bounds for each parameter. If ``None``, uses the
                class ``default_max`` values. Must have length equal to ``dimension``.
            ramp_rate: Custom ramp rate limits for each parameter. If ``None``, uses the
                class ``default_ramp_rate`` values. Must have length equal to ``dimension``.
                Each element can be ``None`` (no limit) or a float (max change per step).
            dtype: NumPy data type for the action arrays (default: ``np.float64``).
                Used for creating action spaces.

        Raises:
            ValueError: If ``name`` class attribute is not defined
            ValueError: If ``dimension`` class attribute is not defined or not a positive integer
            ValueError: If ``config_mapping`` class attribute is not defined
            ValueError: If ``default_min``, ``default_max``, or ``default_ramp_rate`` do not match the dimension
            ValueError: If provided ``min``, ``max``, or ``ramp_rate`` do not match the dimension
        """
        self.dtype = dtype
        # Validate that required class attributes are properly defined
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define 'name' class attribute"
            )

        if not isinstance(self.name, str):
            raise TypeError(
                f"{self.__class__.__name__} 'name' class attribute must be a string"
            )

        if self.dimension is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'dimension' class attribute"
            )

        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(
                f"dimension must be a positive integer, got {self.dimension}"
            )

        if self.config_mapping is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'config_mapping' class attribute"
            )

        if (
            not isinstance(self.default_min, list)
            or len(self.default_min) != self.dimension
        ):
            raise ValueError(f"default_min must be a list of length {self.dimension}")

        if min is not None:
            if not isinstance(min, list) or len(min) != self.dimension:
                raise ValueError(f"min must be a list of length {self.dimension}")

        if (
            not isinstance(self.default_max, list)
            or len(self.default_max) != self.dimension
        ):
            raise ValueError(f"default_max must be a list of length {self.dimension}")

        if max is not None:
            if not isinstance(max, list) or len(max) != self.dimension:
                raise ValueError(f"max must be a list of length {self.dimension}")

        if (
            not isinstance(self.default_ramp_rate, list)
            or len(self.default_ramp_rate) != self.dimension
        ):
            raise ValueError(
                f"default_ramp_rate must be a list of length {self.dimension}"
            )

        if ramp_rate is not None:
            if not isinstance(ramp_rate, list) or len(ramp_rate) != self.dimension:
                raise ValueError(f"ramp_rate must be a list of length {self.dimension}")

        # Initialize minimum and maximum bounds for the action parameters
        # Use provided values or defaults, converting to numpy arrays
        self._min = np.array(
            min if min is not None else self.default_min, dtype=self.dtype
        )
        self._max = np.array(
            max if max is not None else self.default_max, dtype=self.dtype
        )

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
        self.values = np.array(copy.deepcopy(self._min), dtype=self.dtype)

        # Ramp up/down limits - convert None to np.inf for unlimited ramp rates
        ramp_rate_input = ramp_rate if ramp_rate is not None else self.default_ramp_rate
        self.ramp_rate = np.array(
            [r if r is not None else np.inf for r in ramp_rate_input], dtype=self.dtype
        )

        self.dtype = dtype

    @property
    def min(self) -> NDArray[np.floating]:
        """Minimum bounds for this action parameters.

        Returns:
            numpy.ndarray: Array of minimum values, one for each parameter
                controlled by this action.
        """
        return self._min

    @property
    def max(self) -> NDArray[np.floating]:
        """Maximum bounds for this action parameters.

        Returns:
            numpy.ndarray: Array of maximum values, one for each parameter
                controlled by this action.
        """
        return self._max

    def _set_values(self, new_values: list[float] | NDArray[np.floating]) -> None:
        """Set the current parameter values for this action.

        Values are automatically clipped to the action's bounds and ramp rate limits.
        The method ensures all values remain within the defined minimum and maximum
        bounds, and that changes between timesteps do not exceed the ramp rate limits.

        Args:
            new_values: The new parameter values to set. Can be a list or numpy array.
                Must have length equal to the action's dimension.

        Raises:
            ValueError: If the length of the values list does not match the expected dimension.
        """
        if len(new_values) != self.dimension:
            raise ValueError(f"Expected {self.dimension} values, got {len(new_values)}")

        # Clip values to bounds
        clipped_values = np.clip(new_values, self.min, self.max)

        # Apply ramp rate limits (np.inf means no limit)
        ramp_limited_values = np.clip(
            clipped_values, self.values - self.ramp_rate, self.values + self.ramp_rate
        )

        self.values = ramp_limited_values

    def init_dict(self, config_dict: dict[str, Any]) -> None:
        """Initialize a TORAX configuration dictionary with this action parameters.

        This method sets up the configuration dictionary with the action current
        values at ``time=0``, creating the proper time-dependent parameter structure
        expected by TORAX.

        Args:
            config_dict: The TORAX configuration dictionary to initialize.
                Must have the nested structure expected by this action
                config_mapping.

        Raises:
            KeyError: If the configuration dictionary does not have the expected
                structure for this action parameters.
            RuntimeError: If any error occurs during the initialization process.
        """
        try:
            self._apply_mapping(config_dict, time="init")
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while initializing the action in the dictionary: {e}"
            )

    def update_to_config(self, config_dict: dict[str, Any], time: float) -> None:
        """Update a TORAX configuration dictionary with new action values.

        This method updates the time-dependent parameters in the configuration
        dictionary with the action current values at the specified time.
        Scaling factors from ``config_mapping`` are applied consistently.

        Args:
            config_dict: The TORAX configuration dictionary to update.
              Must have been previously initialized with ``init_dict``.
            time: Simulation time for this update. Must be > 0.

        Note:
            The configuration dictionary must have been initialized with
            ``init_dict`` before calling this method. Values are scaled by the
            factors defined in ``config_mapping`` before being stored.
        """
        self._apply_mapping(config_dict, time=time)

    def _apply_mapping(self, config_dict: dict[str, Any], time: float | str) -> None:
        """Apply the action values to a TORAX configuration dictionary.

        This method traverses the configuration dictionary using the paths defined
        in ``config_mapping`` and sets the appropriate values with consistent scaling.
        For ``time="init"`` (initialization), it creates new time-dependent parameter entries.
        For ``time>0``, it updates existing entries. Scaling factors are applied in both cases.

        Args:
            config_dict: The TORAX configuration dictionary to modify
            time: Simulation time. If ``"init"``, initializes new time-dependent parameters.
                If ``>0``, updates existing time-dependent parameters.

        Note:
            This is an internal method used by ``init_dict`` and ``update_to_config``.
            The configuration format follows TORAX conventions where time-dependent
            parameters are stored as ``({time: scaled_value, ...}, "STEP")`` tuples.
            Scaling factors from ``config_mapping`` are consistently applied during
            both initialization and updates.
        """
        for dict_path, (idx, factor) in self.config_mapping.items():
            # drill down into config_dict
            d = config_dict
            for key in dict_path[:-1]:
                d = d[key]

            key = dict_path[-1]
            # Handle the case of the initial condition
            if time == "init":
                # Check if there is no value associated to the existing key
                if d[key] != {}:
                    if isinstance(d[key], (float, int)):  # noqa: UP038
                        self.values[idx] = d[key]
                    elif isinstance(d[key], dict):
                        self.values[idx] = d[key][0]
                    elif isinstance(d[key], tuple) and 0 in d[key][0]:
                        if isinstance(d[key][0], (list, tuple)):  # noqa: UP038
                            pos = d[key][0].index(0)
                        elif isinstance(d[key][0], np.ndarray):
                            pos = np.where(d[key][0] == 0)[0][0]
                        self.values[idx] = d[key][1][pos]
                    # TODO: This log is only valid if we do not do a restart from a .nc file, since in such a case,
                    # no initial condition are needed. Currently it logs it but (somehow, to be investigated) it is
                    # not taken into account.
                    logger.debug(
                        f" using {self.values[idx]} as initial condition for: {key}"
                    )
                else:
                    logger.warning(
                        f" using the lower bound {self.values[idx]} of {key} as initial condition. Consider providing one in the configuration file."
                    )
                # Apply the factor during initialization as well for consistency
                d[key] = ({0: self.values[idx] * factor}, "STEP")
            else:
                d[key][0].update({time: self.values[idx] * factor})

    def get_mapping(self) -> dict[tuple[str, ...], int]:
        """Get the mapping of configuration dictionary paths to action indices and factors.

        Returns:
            dict[tuple[str, ...], tuple[int, float]]: Mapping of config dictionary paths
                to tuples of (action_parameter_index, scaling_factor).
        """
        return self.config_mapping

    def __repr__(self) -> str:
        """Return a string representation of the action.

        Returns:
            str: String showing the action class name, current values, and bounds.
        """
        return f"{self.__class__.__name__}(values={self.values}, min={self.min}, max={self.max})"


class ActionHandler:
    """Internal container and manager for multiple actions.

    This class is used internally by the gymtorax framework to manage collections
    of actions.

    Args:
        actions: List of `Action` instances to manage.

    Attributes:
        actions: Internal dictionary of managed actions indexed by name.
        action_space: Gymnasium `Dict` space representing all managed actions.
        number_of_updates: Counter tracking the number of action updates performed.
    """

    def __init__(self, actions: list[Action]) -> None:
        """Initialize the ActionHandler with a list of actions.

        Validates action compatibility, builds the action space, and sets up
        internal tracking structures.

        Args:
            actions: List of `Action` instances to manage. Actions must have unique
                names and compatible configuration mappings.

        Raises:
            ValueError: If duplicate action names or configuration paths are found,
                or if incompatible actions (e.g., both `Ip` and `Vloop`) are provided.
        """
        self._actions = {a.name: a for a in actions}
        self._actions_names = set([a.name for a in actions])
        self._validate_action_handler()
        self.number_of_updates = 0
        self.action_space = self.build_action_space()

    def get_actions(self) -> dict[str, Action]:
        """Get the dictionary of managed actions.

        Returns:
            dict[str, Action]: Dictionary mapping action names to Action instances
                managed by this handler.
        """
        return self._actions

    def get_action_variables(self) -> dict[str, list[str]]:
        """Get a dictionary of state variables modified by the managed actions.

        Returns:
            dict[str, list[str]]: Dictionary mapping variables categories to lists of
                modified state variable names.
        """
        variables = {}

        for action in self.get_actions().values():
            for cat, var in action.state_var.items():
                if cat not in variables:
                    variables[cat] = []
                if var not in variables[cat]:
                    variables[cat].extend(var)

        return variables

    def update_actions(self, actions: dict[str, NDArray[np.floating]]) -> None:
        """Update the current values of all managed actions.

        This method validates that all provided actions exist in the handler,
        converts values to numpy arrays with correct dtypes, validates bounds,
        and updates each action's internal values using the action's ``_set_values``
        method. The update counter is incremented after successful processing.

        Args:
            actions (dict): Dictionary mapping action names to their new values.
                Keys must correspond to existing action names in this handler.
                Values must be numpy arrays compatible with each action expected format and bounds.

        Raises:
            ValueError: If any action name in the actions dict does not exist
                in this handler's managed actions.
        """
        for action_name, _ in actions.items():
            if action_name not in self._actions_names:
                raise ValueError(
                    f"Action '{action_name}' does not exist in this environment."
                )

        # Ensure all action values are numpy arrays with correct dtype
        actions_np_array = {
            name: np.asarray(values, dtype=self._actions[name].dtype)
            for name, values in actions.items()
        }
        if self.action_space.contains(actions_np_array) is False:
            logger.warning(
                f"updating actions with: a_{self.number_of_updates} = {actions}."
                + f" Action values {actions_np_array} are out of bounds !"
            )
        else:
            logger.debug(
                f" updating actions with: a_{self.number_of_updates} = {actions}"
            )

        for action in self.get_actions().values():
            action._set_values(actions_np_array[action.name])

        self.number_of_updates += 1

    def build_action_space(self) -> spaces.Dict:
        """Build a Gymnasium Dict action space from all managed actions.

        Creates a dictionary-based action space where each key corresponds to
        an action's name and each value is a `Box` space with the action's bounds
        and data type.

        Returns:
            gymnasium.spaces.Dict: Action space structure.
                The action space structure is a dictionnary with action names as keys and
                `Box` spaces as values. Each `Box` space uses the action's min/max
                bounds and dtype for proper numerical handling.
        """
        return spaces.Dict(
            {
                action.name: spaces.Box(
                    low=action.min,
                    high=action.max,
                    dtype=action.dtype,
                )
                for action in self.get_actions().values()
            }
        )

    def _validate_action_handler(self) -> None:
        """Validates the action handler.

        This function performs validation checks:
        1. Verifies that no duplicate configuration paths exist across all actions.
        2. Ensures that action names are unique within the handler.
        3. Enforces TORAX-specific constraints (e.g., ``'Ip'`` and ``'Vloop'`` are mutually exclusive).

        The validation ensures that the action configuration is consistent and compatible
        with TORAX simulation requirements.

        Raises:
            ValueError: If any of the following conditions are detected:
                - Duplicate configuration paths across different actions
                - Duplicate action names
                - Both ``'Ip'`` and ``'Vloop'`` actions present simultaneously
                  (TORAX can only use one current control method)
        """
        seen_keys = set()
        seen_names = set()
        for action in self.get_actions().values():
            for key in action.get_mapping().keys():
                if key in seen_keys:
                    raise ValueError(f"Duplicate action parameter detected: {key}")
                seen_keys.add(key)
            if action.name in seen_names:
                raise ValueError(f"Duplicate action name detected: {action.name}")
            seen_names.add(action.name)

        # Check for exclusive presence of Ip or Vloop actions (through their keys)
        if ("profile_conditions", "v_loop_lcfs") in seen_keys and (
            "profile_conditions",
            "Ip",
        ) in seen_keys:
            raise ValueError("Cannot have both Ip and Vloop actions at the same time.")


# =============================================================================
# Pre-configured Action Examples
# =============================================================================
# The following classes are example implementations of actions used
# in TORAX plasma simulations. Users can use these directly.


class IpAction(Action):
    """Example action for controlling plasma current (Ip).

    This action controls the plasma current parameter in TORAX simulations.
    It is a single-parameter action with non-negative bounds.

    Class Attributes:
        name: ``"Ip"``
        dimension: ``1`` (single parameter)
        default_min: ``[_MIN_IP_AMPS]`` (minimum current per TORAX requirements)
        default_max: ``[numpy.inf]``
        default_ramp_rate: ``[None]``
        config_mapping: Maps to ``('profile_conditions', 'Ip')``
        state_var: ``{'scalars': ['Ip']}`` - directly modifies plasma current scalar

    Action Parameters:
        0: Plasma current (Ip) in Amperes

    Example:
        >>> ip_action = IpAction()
        >>> ip_action._set_values([1.5e6])  # 1.5 MA plasma current
    """

    name = "Ip"
    dimension = 1
    default_min = [_MIN_IP_AMPS]  # TORAX requirements
    default_max = [np.inf]
    default_ramp_rate = [None]
    config_mapping = {("profile_conditions", "Ip"): (0, 1)}
    state_var = {"scalars": ["Ip"]}


class VloopAction(Action):
    """Example action for controlling loop voltage at the last closed flux surface.

    This action controls the loop voltage parameter (v_loop_lcfs) in TORAX
    simulations. It is a single-parameter action with non-negative bounds.

    Class Attributes:
        name: ``"V_loop"``
        dimension: ``1`` (single parameter)
        default_min: ``[0.0]``
        default_max: ``[numpy.inf]``
        default_ramp_rate: ``[None]``
        config_mapping: Maps to ``('profile_conditions', 'v_loop_lcfs')``
        state_var: ``{'scalars': ['v_loop_lcfs']}`` - directly modifies loop voltage scalar

    Action Parameters:
        0: Loop voltage (`v_loop_lcfs`) in Volts

    Example:
        >>> vloop_action = VloopAction()
        >>> vloop_action._set_values([2.5])  # 2.5 V loop voltage
    """

    name = "V_loop"
    dimension = 1
    default_min = [0.0]
    default_max = [np.inf]
    default_ramp_rate = [None]
    config_mapping = {("profile_conditions", "v_loop_lcfs"): (0, 1)}
    state_var = {"scalars": ["v_loop_lcfs"]}


class EcrhAction(Action):
    """Example action for controlling Electron Cyclotron Resonance Heating (ECRH).

    This action controls three ECRH parameters: total power, Gaussian location,
    and Gaussian width of the heating profile.

    Class Attributes:
        name: ``"ECRH"``
        dimension: ``3`` (power, location, width)
        default_min: ``[0.0, 0.0, 0.0]``
        default_max: ``[numpy.inf, numpy.inf, numpy.inf]``
        default_ramp_rate: ``[None, None, None]``
        config_mapping: Maps to ECRH source parameters
        state_var: ``{'scalars': ['P_ecrh_e']}`` -
                     modifies total electron-cyclotron power scalar

    Action Parameters:
        0: Total power (`P_total`) in Watts
        1: Gaussian location (`gaussian_location`) - normalized radius [0,1]
        2: Gaussian width (`gaussian_width`) - profile width parameter

    Example:
        >>> ecrh_action = EcrhAction()
        >>> ecrh_action._set_values([5e6, 0.3, 0.1])  # 5MW, r/a=0.3, width=0.1
    """

    name = "ECRH"
    dimension = 3  # power, location, width
    default_min = [0.0, 0.0, 0.01]
    default_max = [np.inf, 1.0, np.inf]
    default_ramp_rate = [None, None, None]
    config_mapping = {
        ("sources", "ecrh", "P_total"): (0, 1),
        ("sources", "ecrh", "gaussian_location"): (1, 1),
        ("sources", "ecrh", "gaussian_width"): (2, 1),
    }
    state_var = {"scalars": ["P_ecrh_e"]}


class NbiAction(Action):
    """Example action for controlling Neutral Beam Injection (NBI).

    This action controls three NBI parameters: heating power, Gaussian location,
    and Gaussian width of the heating profile. The current drive power is automatically
    calculated from the heating power using a configurable conversion factor.

    Class Attributes:
        name: ``"NBI"``
        dimension: ``3`` (heating power, location, width)
        default_min: ``[0.0, 0.0, 0.01]``
        default_max: ``[numpy.inf, 1.0, numpy.inf]``
        default_ramp_rate: ``[None, None, None]``
        config_mapping: Maps to generic heat and current source parameters in TORAX configuration
        state_var: ``{'scalars': ['P_aux_generic_total']}`` - modifies total auxiliary power scalar

    Attributes:
        nbi_w_to_ma: Conversion factor from heating power (W) to current drive (MA).
            Default is ``1/16e6``, meaning 16MW of heating produces 1MA of current.
        config_mapping: Dynamically created in ``__init__`` to use the specified conversion factor.

    Action Parameters:
        0: Heating power (`generic_heat P_total`) in Watts
        1: Gaussian location (shared by heat and current) - normalized radius [0,1]
        2: Gaussian width (shared by heat and current) - profile width parameter

    Example:
        >>> nbi_action = NbiAction()
        >>> nbi_action._set_values([10e6, 0.4, 0.2])  # 10MW heating, r/a=0.4, width=0.2

        >>> # NBI with custom conversion factor
        >>> nbi_custom = NbiAction(nbi_w_to_ma=1/20e6)  # 20MW per 1MA
        >>> nbi_custom._set_values([20e6, 0.3, 0.15])

        >>> # NBI with current drive disabled
        >>> nbi_heating_only = NbiAction(nbi_w_to_ma=0)
        >>> nbi_heating_only._set_values([15e6, 0.5, 0.1])
    """

    name = "NBI"
    dimension = 3  # heating power, location, width
    default_min = [0.0, 0.0, 0.01]
    default_max = [np.inf, 1.0, np.inf]
    default_ramp_rate = [None, None, None]
    # Note: config_mapping is set in __init__ to allow customizable nbi_w_to_ma
    state_var = {"scalars": ["P_aux_generic_total"]}

    def __init__(self, nbi_w_to_ma=1 / 16e6, **kwargs):
        """Initialize NbiAction with configurable heating-to-current conversion.

        Args:
            nbi_w_to_ma: Conversion factor from heating power (Watts) to current drive (MA).
                Default is ``1/16e6``, meaning 16MW of heating produces 1MA of current drive.
                Set to ``0`` to disable current drive while keeping heating.
            **kwargs: Additional arguments passed to the parent `Action` class
                (``min``, ``max``, ``ramp_rate``, ``dtype``).

        Example:
            >>> # Default conversion (16MW -> 1MA)
            >>> nbi = NbiAction()

            >>> # Custom conversion (20MW -> 1MA)
            >>> nbi = NbiAction(nbi_w_to_ma=1/20e6)

            >>> # Heating only, no current drive
            >>> nbi = NbiAction(nbi_w_to_ma=0)
        """
        # Set the config_mapping with the provided nbi_w_to_ma value
        self.config_mapping = {
            ("sources", "generic_heat", "P_total"): (0, 1),
            ("sources", "generic_current", "I_generic"): (0, nbi_w_to_ma),
            ("sources", "generic_heat", "gaussian_location"): (1, 1),
            ("sources", "generic_heat", "gaussian_width"): (2, 1),
            ("sources", "generic_current", "gaussian_location"): (
                1,
                1,
            ),  # Shared location
            ("sources", "generic_current", "gaussian_width"): (2, 1),  # Shared width
        }

        super().__init__(**kwargs)
        self.nbi_w_to_ma = nbi_w_to_ma
