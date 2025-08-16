"""
Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries,
offering convenient access to common simulation parameters and configuration
management for Gymnasium environments.
"""

from typing import Any
import torax
from torax import ToraxConfig
import gymtorax.action_handler as act
import os

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

        print(config)
        self.config_dict: dict[str, Any] = config
        self.validate()
        print(self.config_dict)
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
        
        #NEED TO VERIFY IF KEYS EXIST
        action_list = self.action_handler.get()
        for a in action_list:
            #Already create the key if it does not exist
            if isinstance(a, act.IpAction):
                self.config_dict['profile_conditions']['Ip'] = (a.get_dict(0), 'STEP')
            
            elif isinstance(a, act.VloopAction):
                self.config_dict['profile_conditions']['v_loop_lcfs '] = (a.get_dict(0), 'STEP')
            
            elif isinstance(a, act.EcrhAction):
                if 'ecrh' not in self.config_dict['sources']:
                    raise KeyError("The source name needs to be in the configuration file")
                list_dic = a.get_dict(0)
                self.config_dict['sources']['ecrh']['P_total'] = (list_dic[0], 'STEP')
                self.config_dict['sources']['ecrh']['gaussian_location'] = (list_dic[1], 'STEP')
                self.config_dict['sources']['ecrh']['gaussian_width'] = (list_dic[2], 'STEP')
            
            elif isinstance(a, act.NbiAction):
                if 'generic_heat' not in self.config_dict['sources'] or  'generic_current' not in self.config_dict['sources']:
                    raise KeyError("The source name needs to be in the configuration file")
                list_dic = a.get_dict(0)   
                self.config_dict['sources']['generic_heat']['P_total'] = (list_dic[0], 'STEP')
                self.config_dict['sources']['generic_heat']['gaussian_location'] = (list_dic[2], 'STEP')
                self.config_dict['sources']['generic_heat']['gaussian_width'] = (list_dic[3], 'STEP')
                self.config_dict['sources']['generic_current']['I_generic'] = (list_dic[1], 'STEP')
                self.config_dict['sources']['generic_current']['gaussian_location'] = (list_dic[2], 'STEP')
                self.config_dict['sources']['generic_current']['gaussian_width'] = (list_dic[3], 'STEP')           
            
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