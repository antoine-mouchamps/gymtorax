"""
TORAX Observation Handler Module.

This module provides an abstract framework for handling observations from TORAX
plasma simulation outputs. Observations represent the current state of the plasma
that can be monitored during the simulation, formatted for use with Gymnasium
reinforcement learning environments.

The module converts TORAX DataTree outputs into structured observation spaces
suitable for machine learning applications, with support for custom variable
selection, bounds specification, and automatic removal of action-controlled
variables to prevent redundancy between action and observation spaces.

Workflow:
    1. Create observation handler with desired variables and bounds
    2. Call set_state_variables() to analyze available TORAX output variables
    3. Call set_action_variables() to specify which variables are controlled by actions
    4. Call set_n_grid_points() to set radial grid resolution
    5. Call build_observation_space() to create the Gymnasium space (one-time only)
    6. Use extract_state_observation() during simulation to get state/observation data

Classes:
    Observation: Abstract base class for building observation spaces from TORAX outputs
    AllObservation: Example implementation that includes all available variables
"""

from abc import ABC
from typing import Any
from gymnasium import spaces
from xarray import DataTree, Dataset

import copy
import numpy as np
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


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
        # "numerics": {
        #     "inner_solver_iterations": {"min": -np.inf, "max": np.inf, "size": 1},
        #     "outer_solver_iterations": {"min": -np.inf, "max": np.inf, "size": 1},
        #     "sawtooth_crash": {"min": -np.inf, "max": np.inf, "size": 1},
        #     "sim_error": {"min": -np.inf, "max": np.inf, "size": 1},
        # },
        "profiles": {
            "area": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "chi_bohm_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "chi_bohm_i": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "chi_gyrobohm_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "chi_gyrobohm_i": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "chi_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "chi_turb_i": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "delta": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "delta_lower": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "delta_upper": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "D_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "ei_exchange": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "elongation": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "F": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "FFprime": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "g0": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g0_over_vpr": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "g1": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g1_over_vpr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g1_over_vpr2": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g2": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g2g3_over_rhon": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "g3": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "Ip_profile": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "Ip_profile_from_geo": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "j_bootstrap": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "j_ecrh": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "j_external": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "j_generic_current": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "j_ohmic": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "j_total": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "magnetic_shear": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "n_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "n_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "n_impurity": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "p_alpha_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_alpha_i": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_cyclotron_radiation_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_ecrh_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_generic_heat_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_generic_heat_i": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_icrh_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_icrh_i": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_impurity_radiation_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "p_ohmic_e": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "Phi": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "pprime": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "pressure_thermal_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "pressure_thermal_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "pressure_thermal_total": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "psi": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "psi_from_geo": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "psi_from_Ip": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "psi_norm": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "q": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "R_in": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "r_mid": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "R_out": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "s_gas_puff": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "s_generic_particle": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "s_pellet": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "sigma_parallel": {"min": -np.inf, "max": np.inf, "size": "rho_cell_norm"},
            "spr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "T_e": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "T_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "v_loop": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "V_turb_e": {"min": -np.inf, "max": np.inf, "size": "rho_face_norm"},
            "volume": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "vpr": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "Z_eff": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "Z_i": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
            "Z_impurity": {"min": -np.inf, "max": np.inf, "size": "rho_norm"},
        },
        "scalars": {
            "a_minor": {"min": -np.inf, "max": np.inf, "size": 1},
            "A_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "A_impurity": {"min": -np.inf, "max": np.inf, "size": 1},
            "B_0": {"min": -np.inf, "max": np.inf, "size": 1},
            "beta_N": {"min": -np.inf, "max": np.inf, "size": 1},
            "beta_pol": {"min": -np.inf, "max": np.inf, "size": 1},
            "beta_tor": {"min": -np.inf, "max": np.inf, "size": 1},
            "dW_thermal_dt": {"min": -np.inf, "max": np.inf, "size": 1},
            "drho": {"min": -np.inf, "max": np.inf, "size": 1},
            "drho_norm": {"min": -np.inf, "max": np.inf, "size": 1},
            "E_aux": {"min": -np.inf, "max": np.inf, "size": 1},
            "E_fusion": {"min": -np.inf, "max": np.inf, "size": 1},
            "fgw_n_e_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "fgw_n_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "H20": {"min": -np.inf, "max": np.inf, "size": 1},
            "H89P": {"min": -np.inf, "max": np.inf, "size": 1},
            "H97L": {"min": -np.inf, "max": np.inf, "size": 1},
            "H98": {"min": -np.inf, "max": np.inf, "size": 1},
            "I_aux_generic": {"min": -np.inf, "max": np.inf, "size": 1},
            "I_bootstrap": {"min": -np.inf, "max": np.inf, "size": 1},
            "I_ecrh": {"min": -np.inf, "max": np.inf, "size": 1},
            "Ip": {"min": -np.inf, "max": np.inf, "size": 1},
            "li3": {"min": -np.inf, "max": np.inf, "size": 1},
            "n_e_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "n_e_min_P_LH": {"min": -np.inf, "max": np.inf, "size": 1},
            "n_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "n_i_line_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "n_i_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_alpha_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_alpha_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_alpha_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_generic_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_generic_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_generic_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_aux_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_bremsstrahlung_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_cyclotron_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_ecrh_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_ei_exchange_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_ei_exchange_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_external_injected": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_icrh_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_icrh_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_icrh_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_LH": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_LH_high_density": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_LH_min": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_ohmic_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_radiation_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_SOL_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_SOL_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "P_SOL_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "Phi_b": {"min": -np.inf, "max": np.inf, "size": 1},
            "Phi_b_dot": {"min": -np.inf, "max": np.inf, "size": 1},
            "q95": {"min": -np.inf, "max": np.inf, "size": 1},
            "Q_fusion": {"min": -np.inf, "max": np.inf, "size": 1},
            "q_min": {"min": -np.inf, "max": np.inf, "size": 1},
            "R_major": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_b": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_2_1_first": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_2_1_second": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_3_1_first": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_3_1_second": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_3_2_first": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_3_2_second": {"min": -np.inf, "max": np.inf, "size": 1},
            "rho_q_min": {"min": -np.inf, "max": np.inf, "size": 1},
            "S_gas_puff": {"min": -np.inf, "max": np.inf, "size": 1},
            "S_generic_particle": {"min": -np.inf, "max": np.inf, "size": 1},
            "S_pellet": {"min": -np.inf, "max": np.inf, "size": 1},
            "S_total": {"min": -np.inf, "max": np.inf, "size": 1},
            "T_e_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "T_i_volume_avg": {"min": -np.inf, "max": np.inf, "size": 1},
            "tau_E": {"min": -np.inf, "max": np.inf, "size": 1},
            "v_loop_lcfs": {"min": -np.inf, "max": np.inf, "size": 1},
            "W_pol": {"min": -np.inf, "max": np.inf, "size": 1},
            "W_thermal_e": {"min": -np.inf, "max": np.inf, "size": 1},
            "W_thermal_i": {"min": -np.inf, "max": np.inf, "size": 1},
            "W_thermal_total": {"min": -np.inf, "max": np.inf, "size": 1},
        },
    }

    DEFAULT_VARIABLES = set(DEFAULT_BOUNDS["profiles"].keys()) | set(DEFAULT_BOUNDS["scalars"].keys())


    def __init__(
        self, 
        variables: dict[str, list[str]]|None = None,
        custom_bounds: dict[str, tuple[float, float]]|None = None,
        exclude: list[str]|None = None,
        dtype: np.dtype = np.float64
    ) -> None:
        """
        Initialize an Observation handler.
        
        Args:
            variables: dictionary specifying which variables to include in the
                observation space. Format: {"profiles": [var_names], "scalars": [var_names]}.
                If None, includes all variables except those in exclude list.
            custom_bounds: dictionary of custom bounds for specific variables,
                format: {var_name: (min_value, max_value)}. Overrides DEFAULT_BOUNDS.
            exclude: list of variable names to exclude from the observation space.
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
        self.variables_to_include = variables
        self.variables_to_exclude = set(exclude or [])
        self.dtype = dtype
        
        # These will be set by required setup methods before building observation space
        self.n_rho = None  # Will be set via set_n_grid_points()
        self.action_variables = None # Will be set via set_action_variables()
        self.state_variables = None # Will be set via set_state_variables()

        # Validate that include and exclude are not both specified
        if self.variables_to_include is not None and self.variables_to_exclude:
            raise ValueError("Cannot specify variables to include and exclude at the same time.")

        # Validate custom bounds - ensure bounds are well-formed
        for var, bounds in self.custom_bounds.items():            
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Custom bounds for variable '{var}' must be a tuple of (min, max).")
            if not all(isinstance(b, (int, float)) for b in bounds):
                raise ValueError(f"Custom bounds for variable '{var}' must be numeric.")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"Custom bounds for variable '{var}' must be (min, max) with min < max.")

        # Initialize bounds with defaults, will be updated with custom bounds later
        # IMPORTANT: Create a deep copy to avoid modifying the class-level DEFAULT_BOUNDS
        self.bounds = copy.deepcopy(self.DEFAULT_BOUNDS)

        # Initialize variables list with defaults, will be filtered later
        # These will be updated in _validate_and_filter_variables() based on actual
        # state variables and user selections
        self.variables = {
            "profiles": list(self.bounds["profiles"].keys()),
            "scalars": list(self.bounds["scalars"].keys())
        }

        # Flag to ensure build_observation_space() is only called once
        self.first_state = True

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

        # Define the mapping from symbolic sizes to actual array dimensions
        self._sizes = {
            "rho_face_norm": n_rho + 1,    # Cell interfaces
            "rho_cell_norm": n_rho,        # Transport cells
            "rho_norm": n_rho + 2          # Cells with boundary conditions
        }

    def _validate_and_filter_variables(self) -> None:
        """
        Validate configuration and finalize variable selection and bounds.
        
        This method performs the complete setup of the observation handler by:
        1. Validating all configuration via _validate()
        2. Determining final variable selection (include/exclude logic)
        3. Applying custom bounds while preserving size information
        4. Converting symbolic sizes to actual array dimensions
        5. Removing action variables from observation space
        6. Logging final variable counts and total dimensions
        
        This method is called automatically by build_observation_space() and
        should not be called directly by users.
        """
        self._validate()
        
        # Determine final variable selection
        if self.variables_to_include is None:
            # Include all variables except excluded ones
            self.variables = {
                cat: [var for var in vars_.keys() if var not in self.variables_to_exclude]
                for cat, vars_ in self.state_variables.items()
            }
        else:
            # Use explicitly provided variables
            self.variables = {
                cat: [var for var in vars_] for cat, vars_ in self.variables_to_include.items()
            }

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
        
        # Update all profile variable sizes
        self.bounds["profiles"] = {
            key: {
                "min": val["min"],
                "max": val["max"], 
                "size": self._sizes[val["size"]]
            } for key, val in self.bounds["profiles"].items()
        }

        # Remove variables affected by the action from the variables tracked
        # for states and observations.
        logger.debug(f"The following variables are removed from the "
                     f"state due to being in the action space:"
                     f"\n{self.action_variables}")
        self._remove_action_variables()

        if logger.isEnabledFor(logging.DEBUG):
            total_dim = 0
            for cat, vars_ in self.variables.items():
                total_dim += sum(self.bounds[cat][var]["size"] for var in vars_)
            logger.debug(f"Validation of all variables done. State space "
                         f"is composed of {len(self.variables['profiles'])} "
                         f"profiles variables, and {len(self.variables['scalars'])} "
                         f"scalars variables, for a total of {total_dim} distinct "
                         f"variables.")

    def _validate(self) -> None:
        """
        Validate all observation handler configuration before building spaces.
        
        This method performs comprehensive validation of:
        - State variables have been set via set_state_variables()
        - Action variables have been set via set_action_variables() 
        - Grid points have been set via set_n_grid_points()
        - All specified variables exist in available state variables
        - Custom bounds are properly specified for existing variables
        
        Automatically adds missing variables to DEFAULT_BOUNDS with infinite
        bounds and logs a warning.
        
        Raises:
            ValueError: If required setup methods haven't been called or
                if specified variables don't exist in the state variables.
        """
        if self.state_variables is None:
            raise ValueError("""State variables must be set before building observation space.
                             Call set_state_variables() first.""")

        # Make sure that all state variables appears in the DEFAULT_VARIABLES list
        all_state_variables = set(self.state_variables["profiles"].keys()) | set(self.state_variables["scalars"].keys())
        missing_vars = all_state_variables - self.DEFAULT_VARIABLES

        if missing_vars:
            # Handle variables present in TORAX output but missing from our DEFAULT_BOUNDS
            # Instead of raising an error, automatically add them with infinite bounds
            for missing_var in missing_vars:
                category = "profiles" if missing_var in self.state_variables["profiles"] else "scalars"
                new_variable = {
                    "min": -np.inf,
                    "max": np.inf,
                    "size": len(self.state_variables[category][missing_var]["data"])
                }
                self.DEFAULT_BOUNDS[category][missing_var] = new_variable
                self.bounds[category][missing_var] = new_variable

            logger.warning(f"State variables is missing from DEFAULT_BOUNDS"
                           f". Bounds (-inf; inf) are assumed:\n"
                           f"{sorted(missing_vars)}")

        # Validate explicitly included variables (if provided)
        if self.variables_to_include is not None:
            for category, var_list in self.variables_to_include.items():
                if category not in ["profiles", "scalars"]:
                    raise ValueError(f"Variable category '{category}' not recognized. Use 'profiles' or 'scalars'.")
                for var in var_list:
                    if var not in all_state_variables:
                        raise ValueError(f"Variable '{var}' not found in state variables.")
            if "profiles" not in self.variables_to_include:
                self.variables_to_include["profiles"] = []
            if "scalars" not in self.variables_to_include:
                self.variables_to_include["scalars"] = []

        # Validate exclude list with variables present in the state
        for var in self.variables_to_exclude:
            if var not in all_state_variables:
                raise ValueError(f"Excluded variable '{var}' not found in available state variables.")

        # Validate custom bounds - ensure bounds appear in the state variables
        for var, _ in self.custom_bounds.items():
            if var not in all_state_variables:
                raise ValueError(f"Custom bound variable '{var}' not found in available state variables.")

        # Ensure action variables have been set
        if self.action_variables is None:
            raise ValueError("""Action variables must be set before building observation space.
                             Call set_action_variables() first.""")

        # Ensure grid points have been set
        if self._sizes is None:
            raise ValueError("""Number of radial grid points (n_rho) must be set before 
                             building observation space. Call set_n_grid_points(n_rho) first.""")

    def set_action_variables(self, variables: dict[str, list[str]]) -> None:
        """
        Set the variables that are controlled by actions.
        
        Action variables will be removed from the observation space to avoid
        including controllable parameters in the state observation, which would
        create redundancy with the action space.
        
        Args:
            variables: Dictionary specifying action variables by category.
                Format: {"profiles": [var_names], "scalars": [var_names]}
                
        Note:
            This method must be called before build_observation_space().
        """
        self.action_variables = variables

    def set_state_variables(self, state: DataTree) -> None:
        """
        Set the state variables available from TORAX simulation output.
        
        This method analyzes the TORAX DataTree to determine which variables
        are actually present in the simulation output. This is important because
        not all variables in DEFAULT_BOUNDS may be available depending on the
        TORAX configuration and models used.
        
        Args:
            state: TORAX DataTree containing the simulation output structure
                with /profiles/ and /scalars/ datasets.
                
        Note:
            This method must be called before build_observation_space().
            Variables missing from the simulation output but present in
            DEFAULT_BOUNDS will be automatically added with infinite bounds.
        """
        self.state_variables = self._get_state_as_dict(state)

    def extract_state_observation(self, datatree: DataTree) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        """
        Extract complete state and filtered observation from TORAX DataTree output.

        This method processes the TORAX simulation output and returns both the
        complete state (all variables selected for this observation handler) and 
        the filtered observation (same as state - maintained for API consistency).

        Args:
            datatree: TORAX simulation output containing profiles and scalars datasets.
                
        Returns:
            tuple containing:
            - state (dict): Complete state with selected variables
                Format: {"profiles": {var: ndarray}, "scalars": {var: scalar}}
            - observation (dict): Same as state (maintained for API consistency)  
                Format: {"profiles": {var: ndarray}, "scalars": {var: scalar}}
                
        Note:
            Both returned dictionaries are identical. This method maintains the
            separate state/observation return values for API consistency, but
            the observation filtering is now handled during space construction
            via action variable removal.
        """
        state = self._get_state_as_dict(datatree)

        # Build complete state dictionary with all available TORAX variables
        state = {
            "profiles": {
                var: state["profiles"][var]["data"][0]
                for var in self.state_variables["profiles"]
            },
            "scalars": {
                var: (state["scalars"][var]["data"][0]
                      if isinstance(state["scalars"][var]["data"], list)
                      else state["scalars"][var]["data"])
                for var in self.state_variables["scalars"]
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

    def _remove_action_variables(self) -> None:
        """
        Remove action variables from the observation handler.
        
        This method removes variables that are controlled by actions from both
        the variables list and bounds dictionary. This prevents action variables
        from appearing in the observation space, avoiding redundancy between
        the action and observation spaces.
        
        Raises:
            KeyError: If action variables reference categories or variables that
                don't exist in the current variable configuration.
        """
        for cat, var_list in self.action_variables.items():
            if cat not in self.variables:
                raise KeyError(f"Variable category '{cat}' not recognized. Use 'profiles' or 'scalars'.")
            for var in var_list:
                if var not in self.variables[cat] or var not in self.bounds[cat]:
                    continue
                    # raise KeyError(f"Variable '{var}' not found in current state variables.")
                self.bounds[cat].pop(var)
                self.variables[cat].remove(var)

    def build_observation_space(self) -> spaces.Dict:
        """
        Build a Gymnasium observation space for the selected variables.
        
        This method creates a nested dict space structure matching the observation
        format, with Box spaces for each variable using the configured bounds.
        
        The method performs one-time initialization by calling _validate_and_filter_variables()
        on the first call, which finalizes variable selection, applies bounds, and
        removes action variables.
        
        Returns:
            spaces.Dict: Gymnasium observation space with the structure:
                {"profiles": dict of Box spaces, "scalars": dict of Box spaces}
                Each Box space has bounds and shape appropriate for the variable.
                
        Raises:
            RuntimeError: If called multiple times (observation variables already set).
            ValueError: If required setup methods (set_n_grid_points, set_action_variables,
                set_state_variables) haven't been called first.
                
        Note:
            This method can only be called once per instance. The setup methods
            set_n_grid_points(), set_action_variables(), and set_state_variables()
            must be called before this method.
        """

        if self.first_state is True:
            self._validate_and_filter_variables()
            self.first_state = False
        else:
            raise RuntimeError("Observation variables have already been set.")

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

    def _get_state_as_dict(self, datatree: DataTree) -> dict[str, dict[str, Any]]:
        """
        Process the DataTree into a structured dictionary format.
        
        This method converts the TORAX DataTree structure into a standardized
        dictionary format that can be easily processed by other methods.
        
        Args:
            datatree: TORAX simulation output DataTree with /profiles/ and /scalars/ datasets.
            
        Returns:
            dict: Structured dictionary with format:
                {
                    "profiles": {var_name: {"data": <numerical_values>, ...}},
                    "scalars": {var_name: {"data": <numerical_values>, ...}}
                }
                
        Note:
            This is an internal method used by set_state_variables() and
            extract_state_observation() to standardize DataTree processing.
        """
        # Extract datasets from the TORAX DataTree structure
        profiles: Dataset = datatree["/profiles/"].ds
        scalars: Dataset = datatree["/scalars/"].ds
 
        # Convert xarray datasets to dictionaries for easier access
        state_profiles, state_scalars = profiles.to_dict(), scalars.to_dict()

        state = {"profiles": state_profiles["data_vars"], "scalars": state_scalars["data_vars"]}

        return state

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
        custom_bounds: dict[str, tuple[float, float]] = None,
        exclude: list[str] = None, 
        dtype: np.dtype = np.float64
    ) -> None:
        """
        Initialize AllObservation with all variables (minus exclusions).
        
        Args:
            custom_bounds: dictionary of custom bounds for specific variables,
                format: {var_name: (min_value, max_value)}.
            exclude: list of variable names to exclude from the observation space.
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
