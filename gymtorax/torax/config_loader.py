import torax 
from torax import ToraxConfig

class ConfigLoader():
    def __init__(self, config:dict):
        self.config_dict:dict = config
        self.config_torax:ToraxConfig = torax.ToraxConfig.from_dict(config)
        
    def get_dict(self) -> dict:
        return self.config_dict

    def get_total_simulation_time(self) -> float:
        return self.config_dict["numerics"]["t_final"]

    def get_simulation_timestep(self) -> float:
        return self.config_dict["numerics"]["fixed_dt"]
    
    def update_config(action):
        """Update the config of the simulation based on the provided action.
        """
        pass