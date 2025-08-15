"""
Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries,
offering convenient access to common simulation parameters and configuration
management for Gymnasium environments.
"""

from typing import Any
import torax
from torax import ToraxConfig
import action_handler as act
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

    def update_config(self, action: list[act.Action], current_time: float, final_time: float, delta_t_a: float) -> None:
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
            raise ValueError("Action must not be empty")
        else:
            for a in action:
                if isinstance(a, act.IpAction):
                    #Ip is a tuple whose first element is a dict and the second is the time interpolation
                    self.config_dict['profile_conditions']['Ip'][0].update(a.get_dict(current_time))
                
                if isinstance(a, act.VloopAction):
                    #Ip is a tuple whose first element is a dict and the second is the time interpolation
                    self.config_dict['profile_conditions']['v_loop_lcfs'][0].update(a.get_dict(current_time))
                
                elif isinstance(a, act.EcrhAction):
                    list_dict = a.get_dict(current_time)
                    self.config_dict['sources']['ecrh']['P_total'][0].update(list_dict[0])
                    self.config_dict['sources']['ecrh']['gaussian_location'][0].update(list_dict[1])
                    self.config_dict['sources']['ecrh']['gaussian_width'][0].update(list_dict[2])
                
                elif isinstance(a, act.NbiAction):
                    list_dict = a.get_dict(current_time)
                    self.config_dict['sources']['generic_heat']['P_total'][0].update(list_dict[0])
                    self.config_dict['sources']['generic_heat']['gaussian_location'][0].update(list_dict[2])
                    self.config_dict['sources']['generic_heat']['gaussian_width'][0].update(list_dict[3])
                    self.config_dict['sources']['generic_current']['I_generic'][0].update(list_dict[1])
                    self.config_dict['sources']['generic_current']['gaussian_location'][0].update(list_dict[2])
                    self.config_dict['sources']['generic_current']['gaussian_width'][0].update(list_dict[3])
                    
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
        
        #Example of how to normalize interpolation for actions
        #We have to repeat this for other actions but let wait util the Action class
        if 'Ip' in self.config_dict['profile_conditions'] and not isinstance(self.config_dict['profile_conditions'].get('Ip'), tuple):
            self.config_dict['profile_conditions']['Ip'] = (self.config_dict['profile_conditions'].get('Ip'), 'STEP')
            print("Warning: we set the time interpolation to 'STEP'.")
            
        elif 'Ip' in self.config_dict['profile_conditions'] and isinstance(self.config_dict['profile_conditions'].get('Ip'), tuple):
            tuple_study = self.config_dict['profile_conditions'].get('Ip')
            if not ('STEP' in tuple_study or 'PIECEWISE_LINEAR' in tuple_study):
                self.config_dict['profile_conditions']['Ip'] = (tuple_study, 'STEP')
                print("Warning: we set the time interpolation to 'STEP'.")
            
            elif 'PIECEWISE_LINEAR' in tuple_study:
                list_temp = list(tuple_study)
                index_ = list_temp.index('PIECEWISE_LINEAR')
                list_temp[index_] = 'STEP'
                
                self.config_dict['profile_conditions']['Ip'] = tuple(list_temp)
                print("Warning: we set the time interpolation to 'STEP'.")
                
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