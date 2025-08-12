"""
Configuration loader for TORAX simulation package.

This module provides a wrapper around TORAX configuration dictionaries,
offering convenient access to common simulation parameters and configuration
management for Gymnasium environments.
"""

from typing import Any
import torax
from torax import ToraxConfig


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
    
    def update_config(self, action) -> None:
        """
        Update the configuration based on the provided action.
        
        TODO
        
        Args:
            action: The action from the RL agent to apply to the configuration
            
        Note:
            This method is currently not implemented and should be customized
            based on the specific action space and configuration parameters
            that need to be modified during training.
        """
        # TODO: Implement based on specific action-to-config mapping
        # This will depend on:
        # - Which TORAX parameters should be controlled by the RL agent
        # - How to map action values to meaningful configuration changes
        pass
