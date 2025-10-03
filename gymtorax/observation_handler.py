"""TORAX Observation Handler Module.

Provides classes for creating observation spaces from TORAX plasma simulation outputs.
Observations represent the current plasma state for use with Gymnasium environments.

The module converts TORAX DataTree outputs into structured observation spaces
with support for variable selection, bounds specification, and automatic
action variable filtering.

Classes:
    `Observation`: Abstract base class for observation space construction.
    `AllObservation`: Implementation that includes all available variables.
"""

import json
import logging
from abc import ABC
from typing import Any

import numpy as np
from gymnasium import spaces
from xarray import Dataset, DataTree

# Set up logger for this module
logger = logging.getLogger(__name__)


class Observation(ABC):
    """Abstract base class for building observation spaces from TORAX DataTree outputs.

    Converts TORAX simulation outputs into structured observation spaces
    for reinforcement learning environments. Handles variable selection,
    bounds specification, and automatic action variable filtering.

    Attributes:
        variables_to_include (dict): Variables to include in observation space.
        variables_to_exclude (dict): Variables to exclude from observation space.
        custom_bounds (dict): Custom bounds for variables.
        dtype (numpy.dtype): Data type for observation arrays.
        action_variables (dict): Variables controlled by actions.
        state_variables (dict): Available variables from TORAX output.
        observation_variables (dict): Final filtered observation variables.
        bounds (dict): Final bounds after processing.
    """

    def __init__(
        self,
        variables: dict[str, list[str]] | None = None,
        custom_bounds_filename: str | None = None,
        exclude: dict[str, list[str]] | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Initialize Observation handler.

        Sets up configuration for the observation handler. Requires subsequent
        calls to set_state_variables(), set_action_variables(), and
        build_observation_space() before use.

        Args:
            variables: Variables to include. Format: ``{"profiles": [names], "scalars": [names]}``.
                If ``None``, includes all available variables except those in ``exclude``.
            custom_bounds_filename: Path to JSON file with custom bounds.
                Format: ``{"profiles": {var: {"min": val, "max": val}}, "scalars": {...}}``.
            exclude: Variables to exclude. Format: ``{"profiles": [names], "scalars": [names]}``.
                Cannot be used with ``variables`` parameter.
            dtype: Data type for observation arrays.

        Raises:
            ValueError: If both ``variables`` and ``exclude`` specified or invalid configuration.
        """
        # Load custom bounds if specified
        if custom_bounds_filename is not None:
            self.custom_bounds = _load_json_file(custom_bounds_filename)
        else:
            self.custom_bounds = {"profiles": {}, "scalars": {}}

        # Validate mutually exclusive parameters
        if variables is not None and exclude:
            raise ValueError(
                "Cannot specify both 'variables' and 'exclude' parameters. "
                "Use either inclusion or exclusion logic."
            )

        self.variables_to_include = variables
        self.variables_to_exclude = exclude

        # Validate exclude parameter structure
        if self.variables_to_exclude is not None:
            for category in self.variables_to_exclude:
                if category not in ["profiles", "scalars"]:
                    raise ValueError(
                        f"Invalid category '{category}'. Use 'profiles' or 'scalars'."
                    )
                for var in self.variables_to_exclude[category]:
                    if not isinstance(var, str):
                        raise ValueError(
                            f"Variable names must be strings, got {type(var)}."
                        )

        self.dtype = dtype

        # These will be set by required setup methods before building observation space
        self.action_variables = None  # Will be set via set_action_variables()
        self.state_variables = None  # Will be set via set_state_variables()

    def _filter_variables(self) -> None:
        """Finalize variable selection and apply bounds configuration.

        Determines final observation variables based on include/exclude logic,
        applies custom bounds, and removes action variables from observation space.
        Called automatically by ``build_observation_space()``.
        """
        if self.variables_to_include is None and self.variables_to_exclude is not None:
            # Include all variables except excluded ones (if they were provided)
            variables_to_exclude = set(self.variables_to_exclude["profiles"]) | set(
                self.variables_to_exclude["scalars"]
            )
            self.observation_variables = {
                cat: [var for var in vars_.keys() if var not in variables_to_exclude]
                for cat, vars_ in self.state_variables.items()
            }
        elif self.variables_to_include is None and self.variables_to_exclude is None:
            # Include all available variables if no include/exclude specified
            self.observation_variables = {
                cat: list(vars_.keys()) for cat, vars_ in self.state_variables.items()
            }
        else:
            # Include only explicitly specified variables
            self.observation_variables = {
                cat: [var for var in vars_]
                for cat, vars_ in self.variables_to_include.items()
            }

        # Remove action variables to prevent observation-action overlap
        logger.debug(f"Removing action variables: {self.action_variables}")
        self._remove_action_variables()

        # Apply custom bounds with default infinite bounds
        self.bounds = {"profiles": {}, "scalars": {}}
        for var in self.observation_variables["profiles"]:
            if var in self.custom_bounds["profiles"]:
                self.bounds["profiles"][var] = {
                    "min": self.custom_bounds["profiles"][var]["min"],
                    "max": self.custom_bounds["profiles"][var]["max"],
                    "size": len(self.state_variables["profiles"][var]["data"][0]),
                }
            else:
                self.bounds["profiles"][var] = {
                    "min": -np.inf,
                    "max": np.inf,
                    "size": len(self.state_variables["profiles"][var]["data"][0]),
                }

        for var in self.observation_variables["scalars"]:
            if var in self.custom_bounds["scalars"]:
                self.bounds["scalars"][var] = {
                    "min": self.custom_bounds["scalars"][var]["min"],
                    "max": self.custom_bounds["scalars"][var]["max"],
                    "size": 1,
                }
            else:
                self.bounds["scalars"][var] = {
                    "min": -np.inf,
                    "max": np.inf,
                    "size": 1,
                }

        # Log final configuration
        if logger.isEnabledFor(logging.DEBUG):
            total_dim = 0
            for cat, vars_ in self.observation_variables.items():
                total_dim += sum(self.bounds[cat][var]["size"] for var in vars_)
            total_obs_bounded = 0
            for cat, vars_ in self.observation_variables.items():
                total_obs_bounded += sum(
                    1 * self.bounds[cat][var]["size"]
                    for var in vars_
                    if self.bounds[cat][var]["min"] > -np.inf
                    or self.bounds[cat][var]["max"] < np.inf
                )
            logger.debug(
                f"Observation space configured:\n"
                f" - Profiles: {len(self.observation_variables['profiles'])}\n"
                f" - Scalars: {len(self.observation_variables['scalars'])}\n"
                f" - Total dimensions: {total_dim}\n"
                f" - Bounded variables: {total_obs_bounded / total_dim:.2%}"
            )

    def _validate(self) -> None:
        """Validate configuration before building observation space.

        Checks that required setup methods have been called and that all
        specified variables exist in the state variables.

        Raises:
            ValueError: If validation fails.
        """
        if self.state_variables is None:
            raise ValueError(
                "State variables not set. Call set_state_variables() first."
            )

        all_state_variables = set(self.state_variables["profiles"].keys()) | set(
            self.state_variables["scalars"].keys()
        )

        # Validate included variables
        if self.variables_to_include is not None:
            for category, var_list in self.variables_to_include.items():
                if category not in ["profiles", "scalars"]:
                    raise ValueError(
                        f"Invalid category '{category}'. Use 'profiles' or 'scalars'."
                    )
                for var in var_list:
                    if var not in all_state_variables:
                        raise ValueError(
                            f"Variable '{var}' not found in state variables."
                        )

            # Ensure both categories exist
            if "profiles" not in self.variables_to_include:
                self.variables_to_include["profiles"] = []
            if "scalars" not in self.variables_to_include:
                self.variables_to_include["scalars"] = []

        # Validate custom bounds
        for category, var_list in self.custom_bounds.items():
            for var in var_list:
                if var not in all_state_variables:
                    raise ValueError(
                        f"Custom bound variable '{var}' not found in state variables."
                    )

        # Ensure action variables have been set
        if self.action_variables is None:
            raise ValueError(
                "Action variables not set. Call set_action_variables() first."
            )

    def set_action_variables(self, variables: dict[str, list[str]]) -> None:
        """Set variables controlled by actions.

        These variables are removed from the observation space to prevent
        redundancy between actions and observations.

        Args:
            variables: Action variables by category.
                Format: ``{"profiles": [names], "scalars": [names]}``.
        """
        self.action_variables = variables

    def set_state_variables(self, state: DataTree) -> None:
        """Set available state variables from TORAX output.

        Catalogs all variables from the TORAX DataTree for inclusion
        in the observation space.

        Args:
            state: TORAX `DataTree` with ``/profiles/`` and ``/scalars/`` datasets.
        """
        self.state_variables = self._get_state_as_dict(state)

    def extract_state_observation(
        self, datatree: DataTree
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        """Extract state and observation data from TORAX output.

        Returns both complete state (all variables) and filtered observation
        (selected variables only).

        Args:
            datatree: TORAX simulation output containing profiles and scalars datasets.

        Returns:
            tuple[dict[str, dict[str, numpy.ndarray]], dict[str, dict[str, numpy.ndarray]]]:
                - state (dict): Complete state with all variables.
                    Format ``{"profiles": {var: array}, "scalars": {var: value}}``
                - observation (dict): Filtered observation with selected variables.
                    Format ``{"profiles": {var: array}, "scalars": {var: value}}``

        """
        state = self._get_state_as_dict(datatree)

        # Extract complete state data
        state = {
            "profiles": {
                var: np.asarray(state["profiles"][var]["data"][0], dtype=self.dtype)
                for var in self.state_variables["profiles"]
            },
            "scalars": {
                var: np.asarray(
                    [state["scalars"][var]["data"][0]]
                    if isinstance(state["scalars"][var]["data"], list)
                    else [state["scalars"][var]["data"]],
                    dtype=self.dtype,
                )
                for var in self.state_variables["scalars"]
            },
        }

        # Filter for observation variables only
        observation = {
            "profiles": {
                var: np.asarray(state["profiles"][var], dtype=self.dtype)
                for var in self.observation_variables["profiles"]
            },
            "scalars": {
                var: np.asarray(state["scalars"][var], dtype=self.dtype)
                for var in self.observation_variables["scalars"]
            },
        }

        return state, observation

    def _remove_action_variables(self) -> None:
        """Remove action variables from the observation handler.

        This method removes variables that are controlled by actions from both
        the variables list and bounds dictionary. This prevents action variables
        from appearing in the observation space, avoiding redundancy between
        the action and observation spaces.

        Variables that are in the action space but not present in the current
        observation variables are silently skipped (no error raised).

        Raises:
            KeyError: If action variables reference categories that do not exist
                (i.e., categories other than 'profiles' or 'scalars').
        """
        for cat, var_list in self.action_variables.items():
            if cat not in self.observation_variables:
                raise KeyError(
                    f"""Variable category '{cat}' not recognized. Use 'profiles' or
                    'scalars'."""
                )
            for var in var_list:
                # Skip variables that are not present in current observation variables
                # This handles cases where action variables may not be in the selected
                # observation set
                if var not in self.observation_variables[cat]:
                    continue
                # Remove the variable from observation variables list
                self.observation_variables[cat].remove(var)

    def build_observation_space(self) -> spaces.Dict:
        """Build Gymnasium observation space for selected variables.

        Creates nested `Dict` space with `Box` spaces for each variable using
        configured bounds. Validates configuration and finalizes variable selection.

        Returns:
            gymnasium.spaces.Dict: Gymnasium `Dict` space with structure
                ``{"profiles": {var: Box}, "scalars": {var: Box}}``

        Raises:
            ValueError: If validation fails or required setup incomplete.
        """
        self._validate()
        self._filter_variables()

        return spaces.Dict(
            {
                "profiles": spaces.Dict(
                    {
                        var: self._make_box(var)
                        for var in self.observation_variables["profiles"]
                    }
                ),
                "scalars": spaces.Dict(
                    {
                        var: self._make_box(var)
                        for var in self.observation_variables["scalars"]
                    }
                ),
            }
        )

    def _make_box(self, var_name: str) -> spaces.Box:
        """Create Gymnasium Box space for variable.

        Args:
            var_name: Variable name.

        Returns:
            gymnasium.spaces.Box: `Box` space with appropriate bounds and shape.
        """
        # Determine variable category and extract bounds
        if var_name in self.bounds["profiles"]:
            low = self.bounds["profiles"][var_name]["min"]
            high = self.bounds["profiles"][var_name]["max"]
            shape = (self.bounds["profiles"][var_name]["size"],)
        else:
            low = self.bounds["scalars"][var_name]["min"]
            high = self.bounds["scalars"][var_name]["max"]
            shape = (self.bounds["scalars"][var_name]["size"],)

        # Create bound arrays
        low_arr = np.full(shape, low, dtype=self.dtype)
        high_arr = np.full(shape, high, dtype=self.dtype)

        return spaces.Box(low=low_arr, high=high_arr, dtype=self.dtype)

    def _get_state_as_dict(self, datatree: DataTree) -> dict[str, dict[str, Any]]:
        """Convert TORAX DataTree to structured dictionary.

        Args:
            datatree: TORAX simulation output.

        Returns:
            dict[str, dict[str, Any]]: Dictionary with format
                ``{"profiles": {var: {"data": values}}, "scalars": {var: {"data": values}}}``
        """
        # Extract datasets
        profiles: Dataset = datatree["/profiles/"].ds
        scalars: Dataset = datatree["/scalars/"].ds

        # Convert to dictionaries
        state_profiles, state_scalars = profiles.to_dict(), scalars.to_dict()

        state = {
            "profiles": state_profiles["data_vars"],
            "scalars": state_scalars["data_vars"],
        }

        return state


def _load_json_file(filename: str) -> dict:
    """Load and validate bounds configuration from JSON file.

    Args:
        filename: Path to JSON bounds file.

    Returns:
        dict: Validated bounds dictionary.

    Raises:
        ValueError: If file format is invalid.
        FileNotFoundError: If file not found.
    """
    with open(filename) as f:
        bounds = json.load(f)

    # Validate and convert string values to numeric
    for cat in bounds:
        if cat not in ["profiles", "scalars"]:
            raise ValueError(f"Invalid category '{cat}'. Use 'profiles' or 'scalars'.")
        for var, var_bounds in bounds[cat].items():
            for bound, val in var_bounds.items():
                if bound not in ["min", "max"]:
                    raise ValueError(
                        f"Invalid bound '{bound}' for '{var}'. Use 'min' or 'max'."
                    )
                if not isinstance(val, int | float | str):
                    raise ValueError(f"Invalid bound value for '{var}': {type(val)}.")
                if val == "inf":
                    var_bounds[bound] = np.inf
                elif val == "-inf":
                    var_bounds[bound] = -np.inf
            if var_bounds["min"] >= var_bounds["max"]:
                raise ValueError(f"Invalid bounds for '{var}': min >= max.")
    return bounds


# =============================================================================
# Pre-configured Observation Examples
# =============================================================================


class AllObservation(Observation):
    """Observation handler that includes all available TORAX variables.

    Creates a complete observation space containing all profile and scalar
    variables available in the TORAX simulation output. Supports variable
    exclusions and custom bounds configuration.

    Example:
        >>> obs = AllObservation()
        >>> obs = AllObservation(exclude={"profiles": ["n_impurity"]})
        >>> obs = AllObservation(custom_bounds_filename="bounds.json")
    """

    def __init__(self, exclude=None, custom_bounds_file=None) -> None:
        """Initialize AllObservation with all available TORAX variables.

        Creates an observation handler that includes all available TORAX variables
        by default, with flexible configuration through keyword arguments.

        Args:
            exclude (dict[str, list[str]] or None): Variables to exclude.
                Format: {"profiles": [names], "scalars": [names]}.
            custom_bounds_file (str or None): Path to JSON file containing
                custom bounds for variables.

        Example:
            >>> obs = AllObservation()
            >>> obs = AllObservation(exclude={"profiles": ["psi"]})
        """
        # Call parent constructor with all variables included by default
        super().__init__(
            variables=None, exclude=exclude, custom_bounds_filename=custom_bounds_file
        )
