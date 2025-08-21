from abc import ABC, abstractmethod
from numpy._typing._array_like import NDArray
from ..torax_wrapper import ToraxApp, ConfigLoader
from ..utils import get_dataset

import gymnasium as gym
from gymnasium import spaces
import xarray as xr
from xarray import DataTree, Dataset

import numpy as np
import logging

from ..action_handler import Action, ActionHandler
from ..observation_handler import Observation, ObservationHandler
from ..logger import setup_logging, get_logger


class BaseEnv(gym.Env, ABC):
    """Gymnasium environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, config:dict|None=None, ratio_a_sim:int|None=None, log_level="warning", logfile=None):
        setup_logging(getattr(logging, log_level.upper()), logfile)

        self.observation_handler = ObservationHandler(self.build_observation_variables())
        self.action_handler = ActionHandler(self.build_action_list())
        self.state: Dataset|None = None # States are saved as a Dataset directly

        self.config: ConfigLoader = ConfigLoader(config)
        
        self.T: float = self.config.get_total_simulation_time() # total time (seconds) of the simulation
        self.delta_t_sim: float = self.config.get_simulation_timestep() # elapsed time between two simulation states
        self.delta_t_a: float = ratio_a_sim * self.delta_t_sim # elapsed time between two actions
        self.current_time: float = 0 # current time (seconds) of the simulation
        self.timestep: int = 0 # current amount of timesteps

        config_loader = ConfigLoader(config, self.action_handler)
        self.torax_app: ToraxApp = ToraxApp(config_loader, self.delta_t_a)

        # Build the action and observation spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def step(self, action):
        truncated = False
        info = {}
        
        state = self.torax_app.get_state_data()

        self.torax_app.update_config(action)
        success = self.torax_app.run()

        # if the simulation did not finish, a terminal state is reached
        if not success:
            self.terminated = True

        next_state = self.torax_app.get_state_data()
        self.state = get_dataset(next_state)
        observation = self.observation_handler.get_observation_values(self.state)

        reward = self.reward(state, action, next_state)
        
        self.current_time += self.delta_t_a
        
        # Update timestep
        self.timestep += 1
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, self.terminated, truncated, info 

    def reset(self, *, seed:int|None = None, options:dict|None = None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)
        
        self.terminated = False
        self.truncated = False
        self.timestep = 0
        self.torax_app.start() # initialise the simulator
        state = self.torax_app.get_state_data() # get the initial state
        self.state = get_dataset(state) 
        observation = self.observation_handler.get_observation_values(self.state) # get the initial observation
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, {}
    
    def reward(self, state, action, next_state)->float:
        pass

    def close(self):
        self.torax_app.close() # finish the simulation
        if self.window is not None:
            # pygame.display.quit()
            # pygame.quit()
            pass
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()  
  
    def _build_action_space(self):
        """
        Build the action space :math:`\mathcal A`.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """
        lower_bounds, upper_bounds = [], []
        for action in self.action_handler.get_actions():
            lower_bounds.extend(action.min)
            upper_bounds.extend(action.max)

        space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float64)

        return space

    def _build_observation_space(self):
        lower_bounds, upper_bounds = [], []
        for observation in self.observation_handler.get_observations():
            lower_bounds.extend(observation.min)
            upper_bounds.extend(observation.max)

        space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float64)

        return space


    def _terminal_state(self):
        pass

    def _render_frame(self):
        # TODO
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     pygame.display.init()
        #     self.window = pygame.display.set_mode(
        #         (self.window_size, self.window_size)
        #     )
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()
        pass


    @abstractmethod
    def build_observation_variables(self)->list[Observation]:
        pass
    
    @abstractmethod
    def build_action_list(self)->list[Action]:
        pass