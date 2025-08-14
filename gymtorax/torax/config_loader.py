"""
Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries,
offering convenient access to common simulation parameters and configuration
management for Gymnasium environments.
"""

from typing import Any
import torax
from torax import ToraxConfig
import os

class ConfigLoader:
    """
    A wrapper class for TORAX configuration management.
    
    This class handles the conversion between Python dictionaries and TORAX's
    internal configuration format, providing convenient access to simulation
    parameters commonly needed in Gymnasium environments.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the configuration loader.
        
        Args:
            config: Dictionary containing TORAX configuration parameters.
                   If None, an empty configuration will be created.
                   
        Raises:
            ValueError: If the configuration dictionary is invalid
            TypeError: If config is not a dictionary or None
        """
        if config is None:
            config = {}
        
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary or None")
            
        self.config_dict: dict[str, Any] = config
        
        try:
            self.config_torax: ToraxConfig = torax.ToraxConfig.from_dict(config)
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

    def update_config(self, action: dict, current_time: float, final_time: float, delta_t_a: float) -> dict:
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

        if not action:
            print("No action provided, returning current config.")
        else:
            keys = action.keys()
            # Unlike other variables, 'sources' and 'profile_conditions' require manual merging.
            # TORAX does not update these profiles automatically; it keeps the last state for the entire simulation.
            # This code merges the existing data (before the current time) with the new data.
            if 'sources' in keys: 
                for source_name, new_source_profile in action['sources'].items():
                    old_source_profile = self.config_dict['sources'].get(source_name, {})
                    merged_source_profile = {}

                    for param, new_val in new_source_profile.items():
                        old_val = old_source_profile.get(param)

                        # Si la valeur est un profil temporel (dict)
                        if isinstance(new_val, dict):
                            merged_val = {}

                            # Conserver ancien < t_current
                            if isinstance(old_val, dict):
                                for t, val in old_val.items():
                                    if float(t) < current_time:
                                        merged_val[t] = val

                            # Ajouter nouveau ≥ t_current
                            for t, val in new_val.items():
                                if float(t) >= current_time:
                                    merged_val[t] = val

                            merged_source_profile[param] = merged_val

                        else:
                            # Valeur scalaire → remplacer directement
                            merged_source_profile[param] = new_val

                    self.config_dict['sources'][source_name] = merged_source_profile

            if 'profile_conditions' in keys:
                if 'Ip' in action['profile_conditions']:
                    new_ip_profile = action['profile_conditions']['Ip']
                    old_ip_profile = self.config_dict['profile_conditions'].get('Ip', {})
                    print(old_ip_profile)
                    merged_ip_profile = {}

                    for t, val in old_ip_profile.items():
                        if float(t) < current_time:
                            merged_ip_profile[t] = val
                    for t, val in new_ip_profile.items():
                        if float(t) >= current_time:
                            merged_ip_profile[t] = val

                    self.config_dict['profile_conditions']['Ip'] = merged_ip_profile

                if 'V_loop' in action['profile_conditions']:
                    new_vloop_profile = action['profile_conditions']['V_loop']
                    old_vloop_profile = self.config_dict['profile_conditions'].get('V_loop', {})
                    merged_vloop_profile = {}

                    for t, val in old_vloop_profile.items():
                        if float(t) < current_time:
                            merged_vloop_profile[t] = val
                    for t, val in new_vloop_profile.items():
                        if float(t) >= current_time:
                            merged_vloop_profile[t] = val

                    self.config_dict['profile_conditions']['V_loop'] = merged_vloop_profile

        # Finally, update the TORAX config accordingly
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

