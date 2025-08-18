"""
Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries,
offering convenient access to common simulation parameters and configuration
management for Gymnasium environments.
"""

import torax

from torax import ToraxConfig
from numpy.typing import NDArray
from typing import Any

import gymtorax.action_handler as act

class ConfigLoader:
    """
    A wrapper class for TORAX configuration management.
    
    This class handles the conversion between Python dictionaries and TORAX's
    internal configuration format, providing convenient access to simulation
    parameters commonly needed in Gymnasium environments.
    """
    
    def __init__(self, config: dict[str, Any],
                 action_handler: act.ActionHandler | None = None,
    ):
        """
        Initialize the configuration loader.
        
        Args:
            config: Dictionary containing TORAX configuration parameters.
            action_handler: An optional ActionHandler instance for managing actions.

        Raises:
            ValueError: If the configuration dictionary is invalid
            TypeError: If config is not a dictionary
        """

        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")
        self.action_handler = action_handler

        self.config_dict: dict[str, Any] = config
        self.validate()
        
        try:
            self.config_torax: ToraxConfig = torax.ToraxConfig.from_dict(self.config_dict)
        except Exception as e:
            raise ValueError(f"Invalid TORAX configuration: {e}")
        
        
    def get_dict(self) -> dict[str, Any]:
        """
        Get the raw configuration dictionary.
        
        Returns:
            The original configuration dictionary
        """
        return self.config_dict.copy()  # Return a copy to prevent external modifications

    def get_total_simulation_time(self) -> float:
        """
        Get the total simulation time in seconds.

        This extracts the :code:`t_final` parameter from the numerics section,
        which defines how long the plasma simulation should run.
        
        Returns:
            Total simulation time in seconds
            
        Raises:
            KeyError: If the configuration doesn't contain the required keys
            TypeError: If the value is not a number
        """
        try:
            t_final = self.config_dict["numerics"]["t_final"]
            if not isinstance(t_final, (int, float)):
                raise TypeError("t_final must be a number")
            return float(t_final)
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def set_total_simulation_time(self, time: float) -> None:
        """
        Set the total simulation time in seconds.

        This updates the :code:`t_final` parameter in the numerics section,
        which defines how long the plasma simulation should run.
        
        Args:
            time: Total simulation time in seconds

        Raises:
            KeyError: If the configuration doesn't contain the required keys
            TypeError: If the value is not a number
        """
        if not isinstance(time, (int, float)):
                raise TypeError("t_final must be a number")
        try:
            self.config_dict["numerics"]["t_final"] = float(time)
            self.config_torax = torax.ToraxConfig.from_dict(self.config_dict)
            
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")


    def get_simulation_timestep(self) -> float:
        """
        Get the simulation timestep in seconds.

        This extracts the :code:`fixed_dt` parameter from the numerics section,
        which defines the time step used in the numerical integration.
        
        Returns:
            Simulation timestep in seconds
            
        Raises:
            KeyError: If the configuration doesn't contain the required keys
            TypeError: If the value is not a number
        """
        try:
            fixed_dt = self.config_dict["numerics"]["fixed_dt"]
            if not isinstance(fixed_dt, (int, float)):
                raise TypeError("fixed_dt must be a number")
            return float(fixed_dt)
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def update_config(self, action_array: NDArray, current_time: float, final_time: float, delta_t_a: float) -> None:
        """Update the configuration of the simulation based on the provided action.
        This method updates the configuration dictionary with new values for sources and profile conditions.
        It also prepares the restart file if necessary. 
        Args:
            action: A dictionary containing the new configuration values for sources and profile conditions.
        Returns:
            The updated configuration dictionary.
        """
        
        self.config_dict['numerics']['t_initial'] = current_time
        self.config_dict['numerics']['t_final'] = current_time + delta_t_a
        if self.config_dict['numerics']['t_final'] > final_time:
            self.config_dict['numerics']['t_final'] = final_time

        self.action_handler.update_actions(action_array)
        actions = self.action_handler.get_actions()

        for action in actions:
            action.update_to_config(self.config_dict, current_time)

        # Update the TORAX config accordingly
        self.config_torax = torax.ToraxConfig.from_dict(self.config_dict)
    
    def validate(self) -> None:
        """
        Validate the configuration dictionary.
        
        This method checks that the configuration contains all required keys
        and that their values are of the expected types for a Gym-TORAX
        environment.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # TODO: Implement validation logic based on Gym-TORAX requirements

        if self.config_dict['time_step_calculator']['calculator_type'] != 'fixed':
            raise ValueError(f"Invalid value: calculator_type should always be 'fixed' when using Gym-TORAX, got {self.config_dict['time_step_calculator']['calculator_type']}")

        if 't_initial' in self.config_dict['numerics'] and self.config_dict['numerics']['t_initial'] != 0.0:
            raise ValueError("The 't_initial' in 'numerics' must be set to 0.0 for the initial configuration.")
        
        if(self.action_handler is not None):
            action_list = self.action_handler.get_actions()
            for a in action_list:
                a.init_dict(self.config_dict)


    def setup_for_simulation(self, file_path: str) -> None:
        """
        Prepare the configuration for a simulation run.
        
        This method sets up the configuration for a simulation, ensuring that
        all necessary parameters are correctly initialized and ready for use.

        """
        
        self.config_dict["restart"] = {
            'filename': file_path,
            'time': 0,
            'do_restart': False, 
            'stitch': True,
        }