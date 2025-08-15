from numpy._typing._array_like import NDArray
from ..torax import ToraxApp, ConfigLoader, expand_sources

import gymnasium as gym
from gymnasium import spaces

import numpy as np


class BaseEnv(gym.Env):
    """Gymnasium environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, config:dict|None=None, ratio_a_sim:int|None=None):
        # TODO I HAVE NO CLUE
        
        self.config: ConfigLoader = ConfigLoader(config)
        self.torax_app: ToraxApp = ToraxApp(config, self.delta_t_a)

        self.T: float = self.config.get_total_simulation_time() # total time (seconds) of the simulation
        self.delta_t_sim: float = self.config.get_simulation_timestep() # elapsed time between two simulation states
        self.delta_t_a: float = ratio_a_sim * self.delta_t_sim # elapsed time between two actions
        self.current_time: float = 0 # current time (seconds) of the simulation
        self.timestep: int = 0 # current amount of timesteps

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
        
        torax_state = self.torax_app.get_state()
        state = self._build_gym_state(torax_state)

        self.torax_app.update_config(action)
        success = self.torax_app.run()

        # if the simulation did not finish, a terminal state is reached
        if not success:
            self.terminated = True
        
        next_torax_state = self.torax_app.get_state()
        next_state = self._build_gym_state(next_torax_state)
        self.state = next_state
        observation = self.torax_app.get_observation()
        
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
        torax_state = self.torax_app.get_state() # get initial state
        self.state = self._build_gym_state(torax_state) # get the corresponding Gym state
        observation = self.torax_app.get_observation() # get the initial observation
        
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
        lower, upper  = self.torax_app.get_action_space()

        space = spaces.Box(low=np.array(lower), high=np.array(upper), dtype=np.float64)

        return space

    def _build_observation_space(self):
        lower_bounds, upper_bounds = [], []

        # TODO define the observation space
        # self.torax_app.get_state_space()
        # use spaces.Dict() but idk how yet
        space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float64)

        return space
    
    def _build_gym_state(self, torax_state):
        """"""
        # TODO: Implement this method to convert the torax_state to a gym state representation.
        return np.array([])

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