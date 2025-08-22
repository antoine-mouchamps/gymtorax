"""
TORAX Observation Handler Module.

This module provides an abstract framework for handling observations from TORAX
plasma simulation outputs. Observations represent the current state of the plasma
that can be monitored during the simulation, formatted for use with Gymnasium
reinforcement learning environments.

The module converts TORAX DataTree outputs into structured observation spaces
suitable for machine learning applications, with support for custom variable
selection, bounds specification, and data type configuration.

Classes:
    Observation: Abstract base class for building observation spaces from TORAX outputs
    AllObservation: Example implementation that includes all available variables

Example:
    Create a custom observation handler:
    
    >>> obs_handler = AllObservation(
    ...     custom_bounds={"T_e": (0.0, 50.0)},  # Temperature bounds in keV
    ...     exclude=["n_impurity"],              # Exclude impurity density
    ...     dtype=np.float64
    ... )
    >>> obs_handler.set_n_grid_points(25)  # Set radial resolution
    >>> obs_space = obs_handler.build_observation_space()
"""

from abc import ABC
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from xarray import DataTree, Dataset


class Observation(ABC):
    """
    Abstract base class for building observation spaces from TORAX DataTree outputs.
    
    This class provides the foundation for converting TORAX simulation outputs into
    structured observation spaces suitable for reinforcement learning environments.
    It handles variable selection, bound specification and data extraction from
    TORAX DataTree structure.
    
    The class maintains a comprehensive catalog of all TORAX variables with their
    default bounds and dimensionality, allowing users to create custom observation
    spaces by selecting subsets of variables and specifying custom bounds, if
    needed.
    
    Attributes:
        DEFAULT_BOUNDS (dict): Master catalog of all TORAX variables with their
            default bounds and size specifications. Variables are categorized as:
            - "profiles": Spatially-resolved variables (functions of radius)
            - "scalars": Global/integrated quantities (single values)
        
        variables (dict): Selected variables for this observation space
        custom_bounds (dict): User-specified bounds overriding defaults
        exclude (set): Variables to exclude from the observation space
        dtype (np.dtype): Data type for observation arrays
        n_rho (int): Number of radial grid points (set via set_n_grid_points)
        bounds (dict): Final bounds after applying custom overrides
    """
    
    # Master catalog of all TORAX variables with default bounds and grid sizing
    DEFAULT_BOUNDS = {
    "profiles": {
        "T_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "T_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "psi": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "v_loop": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "n_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "n_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "n_impurity": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "q": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "magnetic_shear": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "Z_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "Z_impurity": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "Z_eff": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "sigma_parallel": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "j_total": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "Ip_profile": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "chi_turb_i": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "chi_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "D_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "V_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "ei_exchange": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "j_bootstrap": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "p_alpha_i": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_generic_heat_i": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_cyclotron_radiation_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_ecrh_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_alpha_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_generic_heat_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_impurity_radiation_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "p_ohmic_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "j_ecrh": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "j_generic_current": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "pressure_thermal_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "pressure_thermal_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "pressure_thermal_total": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "pprime": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "FFprime": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "psi_norm": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "j_external": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "j_ohmic": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "Phi": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "volume": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "area": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "vpr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "spr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "delta": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "elongation": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g0": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g1": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g2": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g3": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g2g3_over_rhon": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "F": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "R_in": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "R_out": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "Ip_profile_from_geo": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "psi_from_geo": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
        "psi_from_Ip": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "delta_upper": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "delta_lower": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "g0_over_vpr": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
        "g1_over_vpr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "g1_over_vpr2": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        "r_mid": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        # "time": removed
        # "rho_face_norm": removed
        # "rho_cell_norm": removed
        # "rho_norm": removed
    },
    "scalars": {
        "v_loop_lcfs": {"min": -np.inf, "max": np.inf, "size": 1},
        "A_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "A_impurity": {"min": -np.inf, "max": np.inf, "size": 1},
        "Ip": {"min": -np.inf, "max": np.inf, "size": 1},
        "W_thermal_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "W_thermal_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "W_thermal_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "tau_E": {"min": -np.inf, "max": np.inf, "size": 1},
        "H89P": {"min": -np.inf, "max": np.inf, "size": 1},
        "H98": {"min": -np.inf, "max": np.inf, "size": 1},
        "H97L": {"min": -np.inf, "max": np.inf, "size": 1},
        "H20": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_SOL_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_SOL_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_SOL_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_external_injected": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_ei_exchange_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_ei_exchange_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_generic_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_generic_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_aux_generic_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_alpha_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_alpha_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_alpha_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_ohmic_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_bremsstrahlung_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_cyclotron_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_ecrh_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_radiation_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "I_ecrh": {"min": -np.inf, "max": np.inf, "size": 1},
        "I_aux_generic": {"min": -np.inf, "max": np.inf, "size": 1},
        "Q_fusion": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_icrh_e": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_icrh_i": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_icrh_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_LH_high_density": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_LH_min": {"min": -np.inf, "max": np.inf, "size": 1},
        "P_LH": {"min": -np.inf, "max": np.inf, "size": 1},
        "n_e_min_P_LH": {"min": -np.inf, "max": np.inf, "size": 1},
        "E_fusion": {"min": -np.inf, "max": np.inf, "size": 1},
        "E_aux": {"min": -np.inf, "max": np.inf, "size": 1},
        "T_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "T_i_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "n_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "n_i_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "n_e_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "n_i_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "fgw_n_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "fgw_n_e_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
        "q95": {"min": -np.inf, "max": np.inf, "size": 1},
        "W_pol": {"min": -np.inf, "max": np.inf, "size": 1},
        "li3": {"min": -np.inf, "max": np.inf, "size": 1},
        "dW_thermal_dt": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_min": {"min": -np.inf, "max": np.inf, "size": 1},
        "q_min": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_3_2_first": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_3_2_second": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_2_1_first": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_2_1_second": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_3_1_first": {"min": -np.inf, "max": np.inf, "size": 1},
        "rho_q_3_1_second": {"min": -np.inf, "max": np.inf, "size": 1},
        "I_bootstrap": {"min": -np.inf, "max": np.inf, "size": 1},
        "S_gas_puff": {"min": -np.inf, "max": np.inf, "size": 1},
        "S_pellet": {"min": -np.inf, "max": np.inf, "size": 1},
        "S_generic_particle": {"min": -np.inf, "max": np.inf, "size": 1},
        "beta_tor": {"min": -np.inf, "max": np.inf, "size": 1},
        "beta_pol": {"min": -np.inf, "max": np.inf, "size": 1},
        "beta_N": {"min": -np.inf, "max": np.inf, "size": 1},
        "S_total": {"min": -np.inf, "max": np.inf, "size": 1},
        "R_major": {"min": -np.inf, "max": np.inf, "size": 1},
        "a_minor": {"min": -np.inf, "max": np.inf, "size": 1},
        "B_0": {"min": -np.inf, "max": np.inf, "size": 1},
        "Phi_b_dot": {"min": -np.inf, "max": np.inf, "size": 1},
        "Phi_b": {"min": -np.inf, "max": np.inf, "size": 1},
        "drho": {"min": -np.inf, "max": np.inf, "size": 1},
        # "drho_norm": removed
        "rho_b": {"min": -np.inf, "max": np.inf, "size": 1},
        # "time": removed
        # "rho_face_norm": removed
        # "rho_cell_norm": removed
        # "rho_norm": removed
    }
}


    def __init__(
        self, 
        variables: Dict[str, List[str]]|None = None,
        custom_bounds: Dict[str, Tuple[float, float]]|None = None,
        exclude: List[str]|None = None,
        dtype: np.dtype = np.float64
    ) -> None:
        """
        Initialize an Observation handler.
        
        Args:
            variables: Dictionary specifying which variables to include in the
                observation space. Format: {"profiles": [var_names], "scalars": [var_names]}.
                If None, includes all variables except those in exclude list.
            custom_bounds: Dictionary of custom bounds for specific variables,
                format: {var_name: (min_value, max_value)}. Overrides DEFAULT_BOUNDS.
            exclude: List of variable names to exclude from the observation space.
            dtype: NumPy data type for the observation arrays (default: np.float64).
                
        Raises:
            ValueError: If any variable in exclude, custom_bounds, or variables is
                not found in DEFAULT_BOUNDS, or if custom bounds are malformed.
                
        Example:
            >>> obs = Observation(
            ...     variables={"profiles": ["T_e", "T_i"], "scalars": ["Ip", "beta_N"]},
            ...     custom_bounds={"T_e": (0.0, 50.0), "T_i": (0.0, 50.0)},
            ...     dtype=np.float64
            ... )
        """
        # Initialize instance attributes
        self.custom_bounds = custom_bounds or {}
        self.exclude = set(exclude or [])
        self.dtype = dtype
        self.n_rho = None  # Will be set via set_n_grid_points()

        # Get the set of all available variables for validation
        all_variables = set(self.DEFAULT_BOUNDS["profiles"].keys()) | set(self.DEFAULT_BOUNDS["scalars"].keys())

        # Validate exclude list - ensure all excluded variables exist
        for var in self.exclude:
            if var not in all_variables:
                raise ValueError(f"Excluded variable '{var}' not found.")

        # Validate custom bounds - ensure variables exist and bounds are well-formed
        for var, bounds in self.custom_bounds.items():
            if var not in all_variables:
                raise ValueError(f"Custom bound variable '{var}' not found.")
            # Check the bounds format
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Custom bounds for variable '{var}' must be a tuple of (min, max).")
            if not all(isinstance(b, (int, float)) for b in bounds):
                raise ValueError(f"Custom bounds for variable '{var}' must be numeric.")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"Custom bounds for variable '{var}' must be (min, max) with min < max.")

        # Validate explicitly included variables (if provided)
        if variables is not None:
            for category, var_list in variables.items():
                if category not in ["profiles", "scalars"]:
                    raise ValueError(f"Variable category '{category}' not recognized. Use 'profiles' or 'scalars'.")
                for var in var_list:
                    if var not in all_variables:
                        raise ValueError(f"Variable '{var}' not found in available variables.")

        # Determine final variable selection after applying exclusions
        if variables is None:
            # Include all variables except excluded ones
            self.variables = {
                cat: [var for var in vars_.keys() if var not in self.exclude]
                for cat, vars_ in self.DEFAULT_BOUNDS.items()
            }
        else:
            # Use explicitly provided variables, still applying exclusions
            self.variables = {
                cat: [var for var in vars_ if var not in self.exclude]
                for cat, vars_ in variables.items()
            }

        # Initialize bounds with defaults, will be updated with custom bounds later
        # IMPORTANT: Create a deep copy to avoid modifying the class-level DEFAULT_BOUNDS
        self.bounds = copy.deepcopy(self.DEFAULT_BOUNDS)

        # Apply custom bounds by overriding defaults where specified
        for var, custom_bound in self.custom_bounds.items():
            # Create new bound entry with custom min/max but preserving size info
            if var in self.variables["profiles"]:
                original_size = self.bounds["profiles"][var]["size"]
                self.bounds["profiles"][var] = {
                    "min": custom_bound[0],
                    "max": custom_bound[1],
                    "size": original_size
                }
            elif var in self.variables["scalars"]:
                original_size = self.bounds["scalars"][var]["size"]
                self.bounds["scalars"][var] = {
                    "min": custom_bound[0],
                    "max": custom_bound[1],
                    "size": original_size
                }

    def set_n_grid_points(self, n_rho: int) -> None:
        """
        Set the number of radial grid points and update array sizes accordingly.

        This method must be called before building the observation space, as it
        determines the array dimensions for spatially-resolved variables.
        
        The TORAX grid uses three different radial coordinate systems:
        - rho_norm: Cell centers (n_rho + 2 points, includes boundary conditions)
        - rho_cell_norm: Cell centers for transport (n_rho points)  
        - rho_face_norm: Cell faces (n_rho + 1 points)

        Args:
            n_rho: Number of radial transport cells in the simulation grid.
                This determines the resolution of profile variables.
                
        Note:
            This updates the size specifications in self.bounds for all profile
            variables, converting symbolic size names to actual array dimensions.
        """
        if not isinstance(n_rho, int) or n_rho <= 0:
            raise ValueError(f"n_rho must be a positive integer, got {n_rho}")
            
        self.n_rho = n_rho
        
        # Define the mapping from symbolic sizes to actual array dimensions
        sizes = {
            "rho_face_norm": self.n_rho + 1,    # Cell interfaces
            "rho_cell_norm": self.n_rho,        # Transport cells
            "rho_norm": self.n_rho + 2          # Cells with boundary conditions
        }
        
        # Helper function to update size while preserving bounds
        def update_size(bound_dict):
            return {
                "min": bound_dict["min"],
                "max": bound_dict["max"], 
                "size": sizes[bound_dict["size"]]
            }

        # Update all profile variable sizes
        self.bounds["profiles"] = {
            key: update_size(val) for key, val in self.bounds["profiles"].items()
        }

    def extract_state_observation(self, datatree: DataTree) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """
        Extract complete state and filtered observation from TORAX DataTree output.

        This method processes the TORAX simulation output and returns both the
        complete state (all available variables) and the filtered observation 
        (only selected variables for this observation handler).

        Args:
            datatree: TORAX simulation output containing profiles and scalars datasets.
                
        Returns:
            Tuple containing:
            - state (dict): Complete state with all available variables
                Format: {"profiles": {var: ndarray}, "scalars": {var: scalar}}
            - observation (dict): Filtered state with only selected variables  
                Format: {"profiles": {var: ndarray}, "scalars": {var: scalar}}
        """
        # Extract datasets from the TORAX DataTree structure
        profiles: Dataset = datatree["/profiles/"].ds
        scalars: Dataset = datatree["/scalars/"].ds

        # Remove singleton dimensions (e.g., time dimension if present)
        profiles = profiles.map(lambda da: da.squeeze())
        scalars = scalars.map(lambda da: da.squeeze())
        
        # Convert xarray datasets to dictionaries for easier access
        state_profiles, state_scalars = profiles.to_dict(), scalars.to_dict()
        
        # Build complete state dictionary with all available TORAX variables
        state = {
            "profiles": {
                var: state_profiles["data_vars"][var]["data"] 
                for var in self.bounds["profiles"].keys()
            },
            "scalars": {
                var: state_scalars["data_vars"][var]["data"] 
                for var in self.bounds["scalars"].keys()
            },
        }
        
        # Filter state to create observation with only selected variables
        observation = {
            "profiles": {
                var: state["profiles"][var] 
                for var in self.variables["profiles"]
            },
            "scalars": {
                var: state["scalars"][var] 
                for var in self.variables["scalars"] 
            }
        }

        return state, observation

    def update_variables(self, variables: dict[str, list[str]]) -> None:
        """
        Remove specified variables from the observation handler.

        This method removes variables from both the observation and state
        variables list and their corresponding bounds.

        Args:
            variables: Dictionary mapping variable categories to lists of variable names
                to remove.
        
        Raises:
            KeyError: If a variable category is not recognized (must be 'profiles' 
                or 'scalars'), or if a specified variable is not found in the 
                current state/observation variables.
        """
        for cat, var_list in variables.items():
            if cat not in self.variables:
                raise KeyError(f"Variable category '{cat}' not recognized. Use 'profiles' or 'scalars'.")
            for var in var_list:
                if var not in self.variables[cat] or var not in self.bounds[cat]:
                    raise KeyError(f"Variable '{var}' not found in current state variables.")
                self.bounds[cat].pop(var)
                self.variables[cat].remove(var)

    def build_observation_space(self) -> spaces.Dict:
        """
        Build a Gymnasium observation space for the selected variables.
        
        This method creates a nested Dict space structure matching the observation
        format, with Box spaces for each variable using the configured bounds.
        
        Returns:
            spaces.Dict: Gymnasium observation space with the structure:
                {"profiles": Dict of Box spaces, "scalars": Dict of Box spaces}
                Each Box space has bounds and shape appropriate for the variable.
                
        Raises:
            ValueError: If n_rho has not been set via set_n_grid_points().
        """
        if self.n_rho is None:
            raise ValueError(
                "Number of radial grid points (n_rho) must be set before building the observation space. "
                "Call set_n_grid_points(n_rho) first."
            )
        
        return spaces.Dict({
            "profiles": spaces.Dict({
                var: self._make_box(var)
                for var in self.variables["profiles"]
            }),
            "scalars": spaces.Dict({
                var: self._make_box(var)
                for var in self.variables["scalars"]
            }),
        })
    
    def _make_box(self, var_name: str) -> spaces.Box:
        """
        Create a Box space for a single variable with appropriate bounds and shape.
        
        Args:
            var_name: Name of the variable to create a Box space for.
            
        Returns:
            spaces.Box: Box space with bounds and shape for the variable.
        """
        # Check if variable is in profiles category
        if var_name in self.bounds["profiles"]:
            low = self.bounds["profiles"][var_name]["min"]
            high = self.bounds["profiles"][var_name]["max"]
            shape = (self.bounds["profiles"][var_name]["size"],)
        # Otherwise, it must be in scalars category
        else:
            low = self.bounds["scalars"][var_name]["min"]
            high = self.bounds["scalars"][var_name]["max"]
            shape = (self.bounds["scalars"][var_name]["size"],)

        # Create arrays for bounds with the appropriate shape and dtype
        low_arr = np.full(shape, low, dtype=self.dtype)
        high_arr = np.full(shape, high, dtype=self.dtype)
        
        return spaces.Box(low=low_arr, high=high_arr, dtype=self.dtype)


# =============================================================================
# Pre-configured Observation Examples
# =============================================================================


class AllObservation(Observation):
    """
    Example observation handler that includes all available TORAX variables.
    
    The observation space will contain all profile and scalar variables available
    in TORAX output, providing complete visibility into the plasma state.
    
    Example:
        >>> # Include everything except impurity-related variables
        >>> obs = AllObservation(exclude=["n_impurity", "Z_impurity", "A_impurity"])
        >>> 
        >>> # Custom bounds for temperatures (0-50 keV range)
        >>> obs_bounded = AllObservation(
        ...     custom_bounds={"T_e": (0.0, 50.0), "T_i": (0.0, 50.0)},
        ...     dtype=np.float64
        ... )
    """

    def __init__(
        self, 
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        exclude: Optional[List[str]] = None, 
        dtype: np.dtype = np.float64
    ) -> None:
        """
        Initialize AllObservation with all variables (minus exclusions).
        
        Args:
            custom_bounds: Dictionary of custom bounds for specific variables,
                format: {var_name: (min_value, max_value)}.
            exclude: List of variable names to exclude from the observation space.
            dtype: NumPy data type for observation arrays.
        """
        # Call parent constructor with variables=None to include all variables
        # The parent class will handle exclusions and custom bounds
        super().__init__(
            variables=None,  # Include all variables
            custom_bounds=custom_bounds,
            exclude=exclude,
            dtype=dtype
        )


if __name__ == "__main__":
    # Example usage
    from torax._src.output_tools import output
    dt = output.load_state_file('gymtorax/output.nc')
    # ds = get_dataset(dt)

    ob = AllObservation()
    obs = ob.extract_state_observation(dt)
    print(obs)

    # def print_dtree(dt, indent=0):
    #     prefix = "  " * indent
    #     print(f"{prefix}{dt.name}/")
    #     if dt.has_data:
    #         for var in dt.ds.variables:
    #             print(f"{prefix}  {var} : {dt.ds[var].dims} -> {dt.ds[var].shape}")
    #     for child in dt.children.values():
    #         print_dtree(child, indent + 1)

    # print_dtree(dt)
    
    # oh = ObservationHandler()
    # os = oh.get_observation(ds)
