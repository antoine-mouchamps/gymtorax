"""Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries, offering
convenient access to common simulation parameters and configuration management for
Gymnasium environments.
"""

from typing import Any

import torax
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

    def get_initial_simulation_time(self, restart=False) -> float:
        """Get the initial simulation time in seconds.

        This extracts the ``t_initial`` parameter from the numerics section,
        which defines the initial time for the plasma simulation.

        Returns:
            Total simulation time in seconds

        Raises:
            KeyError: If the configuration does not contain the required keys
            TypeError: If the value is not a number
        """
        if restart is False:
            if "t_initial" not in self.config_dict["numerics"]:
                t_initial = 0.0
            else:
                t_initial = self.config_dict["numerics"]["t_initial"]
        else:
            t_initial = self.config_dict["restart"]["time"]

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

    def update_config(self, action, current_time: float, delta_t_a: float) -> None:
        """Update the simulation configuration with new timing and action parameters.

        This method updates the TORAX configuration with new time boundaries and
        applies the provided action through the action handler. It handles time
        stepping and rebuilds the TORAX config.

        Args:
            action: Action values to be applied through the action handler.
            current_time: The current simulation time in seconds.
            delta_t_a: The action duration/time step in seconds.

        Raises:
            ValueError: If Ip control is requested but Ip_from_parameters is False.
        """
        self.config_dict["numerics"]["t_initial"] = current_time
        self.config_dict["numerics"]["t_final"] = current_time + delta_t_a

        # TODO: why is this here ???
        # Allow the control of Ip after initialization
        if "Ip" in self.action_handler.get_action_variables():
            if (
                "Ip_from_parameters" in self.config_dict["geometry"]
                and self.config_dict["geometry"]["Ip_from_parameters"] is False
            ):
                raise ValueError(
                    "Control over Ip implies that 'Ip_from_parameters' must be True so that TORAX considers it."
                    + " that TORAX considers it."
                )

        self.action_handler.update_actions(action)
        actions = self.action_handler.get_actions().values()

        for action in actions:
            action.update_to_config(self.config_dict, current_time)

        # Update the TORAX config accordingly
        self.config_torax = torax.ToraxConfig.from_dict(self.config_dict)

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
        environment.

        Raises:
            ValueError: If the configuration is invalid
        """
        action_list = self.action_handler.get_actions().values()
        for a in action_list:
            a.init_dict(self.config_dict)

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
