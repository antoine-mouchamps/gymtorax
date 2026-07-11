"""Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries, offering
convenient access to common simulation parameters and configuration management for
Gymnasium environments.
"""

from typing import Any

import torax
import xarray as xr
from torax import ToraxConfig

from ..action_handler import ActionHandler


class ConfigLoader:
    """A wrapper class for TORAX configuration management.

    This class handles the conversion between Python dictionaries and TORAX's internal
    configuration format, providing convenient access to simulation parameters commonly
    needed in Gymnasium environments.
    """

    def __init__(
        self,
        config: dict[str, Any],
        action_handler: ActionHandler,
    ):
        """Initialize the configuration loader.

        Args:
            config: Dictionary containing TORAX configuration parameters.
            action_handler: `ActionHandler` instance for managing actions.

        Raises:
            ValueError: If the configuration dictionary is invalid
            TypeError: If config is not a dictionary
        """
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")
        self.action_handler = action_handler

        self.config_dict: dict[str, Any] = config
        self._validate()
        try:
            self.config_torax: ToraxConfig = torax.ToraxConfig.from_dict(
                self.config_dict
            )
        except Exception as e:
            raise ValueError(f"Invalid TORAX configuration: {e}")

    def get_dict(self) -> dict[str, Any]:
        """Get the raw configuration dictionary.

        Returns:
            The original configuration dictionary
        """
        return (
            self.config_dict.copy()
        )  # Return a copy to prevent external modifications

    def get_total_simulation_time(self) -> float:
        """Get the total simulation time in seconds.

        This extracts the ``t_final`` parameter from the numerics section,
        which defines how long the plasma simulation should run.

        Returns:
            Total simulation time in seconds

        Raises:
            KeyError: If the configuration does not contain the required keys
            TypeError: If the value is not a number
        """
        try:
            t_final = self.config_dict["numerics"]["t_final"]
            if not isinstance(t_final, int | float):
                raise TypeError("t_final must be a number")
            return float(t_final)
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def set_total_simulation_time(self, time: float) -> None:
        """Set the total simulation time in seconds.

        This updates the ``t_final`` parameter in the numerics section,
        which defines how long the plasma simulation should run.

        Args:
            time: Total simulation time in seconds

        Raises:
            KeyError: If the configuration does not contain the required keys
            TypeError: If the value is not a number
        """
        if not isinstance(time, int | float):
            raise TypeError("t_final must be a number")
        try:
            self.config_dict["numerics"]["t_final"] = float(time)
            self.config_torax = torax.ToraxConfig.from_dict(self.config_dict)

        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def get_initial_simulation_time(self, restart: bool = False) -> float:
        """Get the initial simulation time in seconds.

        This extracts the ``t_initial`` parameter from the numerics section,
        which defines the initial time for the plasma simulation. Defaults to
        ``0.0`` if not set, in accordance with TORAX settings. If ``restart``
        is ``True``, the time is instead read from the ``restart`` section:
        the simulation continues from a previous TORAX run saved as a file.

        Args:
            restart: If ``True``, return the restart time from the ``restart``
                section instead of ``numerics.t_initial``.

        Returns:
            Initial simulation time in seconds

        Raises:
            KeyError: If ``restart`` is ``True`` but the configuration has no
                ``restart`` section with a ``time`` entry
            TypeError: If the value is not a number
        """
        if restart:
            t_initial = self.config_dict["restart"]["time"]
        else:
            t_initial = self.config_dict["numerics"].get("t_initial", 0.0)

        if not isinstance(t_initial, int | float):
            raise TypeError("t_initial must be a number")

        return float(t_initial)

    def get_simulation_timestep(self) -> float:
        """Get the simulation timestep in seconds.

        This extracts the ``fixed_dt`` parameter from the numerics section,
        which defines the time step used in the numerical integration.

        Returns:
            Simulation timestep in seconds

        Raises:
            KeyError: If the configuration does not contain the required keys
            TypeError: If the value is not a number
        """
        try:
            fixed_dt = self.config_dict["numerics"]["fixed_dt"]
            if not isinstance(fixed_dt, int | float):
                raise TypeError("fixed_dt must be a number")
            return float(fixed_dt)
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def get_n_grid_points(self) -> int:
        """Get the number of radial grid points (rho) in the simulation.

        This extracts the ``n_rho`` parameter from the geometry section,
        which defines the number of radial grid points in the simulation. If
        the parameter is not set, a default value of ``25`` will be used, in
        accordance to TORAX settings.

        Returns:
            Number of radial grid points (rho)

        Raises:
            TypeError: If the value is not an integer
        """
        if "n_rho" in self.config_dict["geometry"]:
            n_rho = self.config_dict["geometry"]["n_rho"]
            if not isinstance(n_rho, int):
                raise TypeError("n_rho must be an integer")
            return n_rho
        else:
            return 25

    def apply_action(
        self, action, current_time: float, delta_t_a: float
    ) -> dict[str, Any]:
        """Apply a new action and collect its runtime-parameter updates.

        This method updates the action handler with the new action values
        (applying bounds and ramp-rate clipping) and returns the corresponding
        per-window parameter ramps. It does NOT touch ``config_dict`` or
        ``config_torax``: after initialization, the configuration only
        describes the start of the episode, and per-step changes are applied
        in place on the step function's `RuntimeParamsProvider`, which is much
        cheaper than a config rebuild and cannot change the JAX pytree
        structure (no recompilation).

        Args:
            action: Action values to be applied through the action handler.
            current_time: The current simulation time in seconds.
            delta_t_a: The action duration/time step in seconds.

        Returns:
            Mapping of dot-separated runtime-parameter paths to update values
            (action ramps only; the caller adds the ``numerics.t_final``
            window bound), suitable for
            ``RuntimeParamsProvider.update_provider_from_mapping``.
        """
        self.action_handler.update_actions(action)

        provider_updates: dict[str, Any] = {}
        for act in self.action_handler.get_actions().values():
            provider_updates |= act.get_provider_updates(current_time, delta_t_a)

        return provider_updates

    def get_current_action_values(self) -> dict[str, Any]:
        """Get the current action values from the action handler.

        Returns:
            Dictionary of current action values
        """
        current_values = {}
        for action in self.action_handler.get_actions().values():
            current_values[action.name] = action.values

        return current_values

    def _validate(self) -> None:
        """Validate the configuration dictionary.

        This method checks that the configuration contains all required keys
        and that their values are of the expected types for a Gym-TORAX
        environment, and initializes the action parameters in the
        configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid
            RuntimeError: If an action cannot be initialized in the
                configuration dictionary
        """
        # Controlling Ip requires TORAX to take Ip from the parameters rather
        # than from the geometry file.
        if "Ip" in self.action_handler.get_action_variables().get("scalars", []):
            if self.config_dict["geometry"].get("Ip_from_parameters") is False:
                raise ValueError(
                    "Control over Ip implies that 'Ip_from_parameters' must be"
                    " True so that TORAX considers it."
                )

        action_list = self.action_handler.get_actions().values()
        for a in action_list:
            a.init_dict(self.config_dict)

        # When restarting from a previous run, the action initial values
        # in the configuration file must match the action values
        # recorded in the last state of the restart file.
        if self.config_dict.get("restart", {}).get("do_restart", False):
            restart = self.config_dict["restart"]

            # TORAX restores psi from the restart file through
            # profile_conditions, which is only honored when initial_psi_mode
            # is 'profile_conditions' (the TORAX default). Any other mode
            # silently recomputes the initial psi instead of restoring it,
            # making the restarted plasma state inconsistent with the file.
            psi_mode = self.config_dict.get("profile_conditions", {}).get(
                "initial_psi_mode", "profile_conditions"
            )
            if psi_mode != "profile_conditions":
                raise ValueError(
                    f"Restarting with initial_psi_mode='{psi_mode}' would "
                    "silently recompute the initial psi instead of restoring "
                    "it from the restart file. Remove initial_psi_mode from "
                    "the configuration (or set it to 'profile_conditions') "
                    "when restarting."
                )
            try:
                with open(restart["filename"], "rb") as f:
                    dt_open = xr.open_datatree(f)
                    data_tree = dt_open.compute()
            except Exception as e:
                raise ValueError(
                    f"Could not load restart file '{restart.get('filename', 'no filename')}' for "
                    f"the restart consistency check: {e}"
                )

            data_tree = data_tree.sel(time=restart["time"], method="nearest")
            scalars = data_tree.children["scalars"].dataset.squeeze()
            self.action_handler.validate_restart_scalars(scalars)

    def validate_discretization(self, discretization_torax: str) -> None:
        """Validate the discretization settings.

        This method checks that the discretization settings are consistent
        and valid for the simulation.

        Raises:
            ValueError: If the discretization settings are invalid
        """
        if discretization_torax == "fixed":
            if "calculator_type" in self.config_dict["time_step_calculator"]:
                if (
                    self.config_dict["time_step_calculator"]["calculator_type"]
                    != "fixed"
                ):
                    raise ValueError(
                        "calculator_type must be set to 'fixed' for fixed discretization."
                        " for fixed discretization."
                    )
        elif discretization_torax == "auto":
            if "calculator_type" in self.config_dict["time_step_calculator"]:
                if (
                    self.config_dict["time_step_calculator"]["calculator_type"]
                    == "fixed"
                ):
                    raise ValueError(
                        "calculator_type must not be set to 'fixed' for auto discretization."
                        " for auto discretization."
                    )
        else:
            raise ValueError("Invalid discretization_torax setting.")
