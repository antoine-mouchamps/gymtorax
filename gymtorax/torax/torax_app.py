from dataclasses import dataclass

@dataclass
class Bounds:
    min: float
    max: float

@dataclass
class SourceBounds:
    total: Bounds
    loc: Bounds
    width: Bounds
    
def expand_sources(ES_k_bounds:list[SourceBounds]) -> list[Bounds]:
    """Transform the list of SourceBounds (tuples of Bounds) into the corresponding
    list of bounds (deleting the SourceBounds wrapper)

    Returns
    -------
    list
        The expanded action space for the sources.
    """
    return [bounds for source in ES_k_bounds for bounds in (source.total, source.loc, source.width)]
    
class ToraxApp:
    """Takes care of the Torax application lifecycle and state management."""
    def __init__(self, config: dict):
        self.config = config
        self.started: bool = False

    def start(self):
        """Initialize the Torax application with the provided configuration.
        """
        self.started = True
        
    def close(self):
        """Close TORAX ? DELETE OUTPUT FILE(s)"""
        pass

    def update_config(action):
        """Update the config of the simulation based on the provided action.
        """
        # self.config.update_config()
        pass

    def run(self, config_dict: dict) -> bool:
        """Perform a single simulation step inside of TORAX"""

        pass

    def get_action_space(self) -> tuple[Bounds, Bounds, list[SourceBounds]]:
        """Get the action space for the simulation.
        Format is (Ip_bounds, Vloop_bounds, ES_k_bounds), with Ip_bounds and
        Vloops_bounds being a tuple of (min, max), and ES_k_bounds being a
        list of sources with ([min,max], [min,max], [min,max]) bounds.
        """
        pass
    
    def get_state_space(self) -> list[Bounds]:
        """Get the state space for the simulation.
        
        """
        pass
    
    def get_state(self):
        """_summary_
        """
        pass
    
    def get_observation(self):
        "Return the observation of the last simulated state"
        pass
    
