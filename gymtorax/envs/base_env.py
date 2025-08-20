from abc import ABC, abstractmethod
from numpy._typing._array_like import NDArray
from ..torax_wrapper import ToraxApp, ConfigLoader

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from ..action_handler import Action, ActionHandler
from ..observation_handler import Observation


class BaseEnv(gym.Env, ABC):
    """Gymnasium environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, config:dict|None=None, discretization_torax: str="auto", ratio_a_sim: int=None, delta_t_a: int=None):
        self.observation_handler = self.build_observation_variables()
        self.action_handler = ActionHandler(self.build_action_list())
        self.state: dict|None = None

        self.config: ConfigLoader = ConfigLoader(config)
        self.config.validate_discretization(discretization_torax)
        
        self.T: float = self.config.get_total_simulation_time() # total time (seconds) of the simulation
        if discretization_torax == "auto":
            if delta_t_a is None:
                raise ValueError("delta_t_a must be provided for auto discretization")
            self.delta_t_a: float = delta_t_a # elapsed time between two actions
        elif discretization_torax == "fixed":
            if ratio_a_sim is None:
                raise ValueError("ratio_a_sim must be provided for fixed discretization")
            delta_t_sim: float = self.config.get_simulation_timestep() # elapsed time between two simulation states
            self.delta_t_a: float = ratio_a_sim * delta_t_sim # elapsed time between two actions
        else:
            raise TypeError("Invalid type for ratio_a_sim")

        self.current_time: float = 0 # current time (seconds) of the simulation
        self.timestep: int = 0 # current amount of timesteps

        config_loader = ConfigLoader(config, self.action_handler)
        self.torax_app: ToraxApp = ToraxApp(config_loader, self.delta_t_a)

        # Build the action and observation spaces
        self.action_space = self._build_action_space()
        self.observation_handler.set_n_grid_points(self.config.get_n_grid_points())
        self.observation_space = self.observation_handler.build_observation_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, *, seed:int|None = None, options:dict|None = None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)
        
        self.terminated = False
        self.truncated = False
        self.timestep = 0
        self.torax_app.start() # initialise the simulator
        torax_state = self.torax_app.get_state_data() # get the initial state
        self.state, self.observation = self.observation_handler.extract_state_observation(torax_state) # get the initial observation

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, {}

    def step(self, action):
        truncated = False
        info = {}
        
        state = self.torax_app.get_state_data()

        self.torax_app.update_config(action)
        success = self.torax_app.run()

        # if the simulation did not finish, a terminal state is reached
        if not success:
            self.terminated = True

        next_torax_state = self.torax_app.get_state_data()
        next_state, observation = self.observation_handler.extract_state_observation(next_torax_state)
        self.state, self.observation = next_state, observation

        reward = self.reward(state, next_state, action)
        
        self.current_time += self.delta_t_a
        
        # Update timestep
        self.timestep += 1
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, self.terminated, truncated, info 

    def reward(self, state, next_state, action) -> float:
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
    def build_observation_variables(self)->Observation:
        pass
    
    @abstractmethod
    def build_action_list(self)->list[Action]:
        pass