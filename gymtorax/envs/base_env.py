"""
TORAX Base Environment Module.

This module provides the abstract base class for TORAX plasma simulation environments
compatible with the Gymnasium reinforcement learning framework. It integrates TORAX
physics simulations with RL interfaces, handling time discretization, action/observation
spaces, and the simulation lifecycle.

The BaseEnv class serves as a foundation for creating specific plasma control tasks by:
- Managing TORAX configuration and simulation execution
- Defining action and observation space structures  
- Handling time discretization and episode management
- Providing hooks for custom reward functions and terminal conditions
- Configurable logging system for debugging and monitoring

Classes:
    BaseEnv: Abstract base class for TORAX Gymnasium environments

Example:
    Create a custom environment by extending BaseEnv:
    
    >>> class PlasmaControlEnv(BaseEnv):
    ...     def build_observation_variables(self):
    ...         return AllObservation(exclude=["n_impurity"])
    ...     
    ...     def build_action_list(self):
    ...         return [IpAction(), EcrhAction()]
    ...     
    ...     def reward(self, state, next_state, action):
    ...         # Custom reward logic
    ...         return -abs(next_state["scalars"]["beta_N"] - 2.0)
"""

from abc import ABC, abstractmethod
from typing import Any
from numpy._typing._array_like import NDArray

import numpy as np
import gymnasium as gym
import logging

from ..action_handler import Action, ActionHandler
from ..observation_handler import Observation
from ..torax_wrapper import ToraxApp, ConfigLoader
from ..logger import setup_logging
import gymtorax.rendering.visualization as viz

# Set up logger for this module
logger = logging.getLogger(__name__)


class BaseEnv(gym.Env, ABC):
    """
    Abstract base class for TORAX plasma simulation environments.
    
    This class integrates TORAX physics simulations with the Gymnasium reinforcement
    learning framework, providing a standardized interface for plasma control tasks.
    It handles the complexities of time discretization, simulation management, and
    action/observation space construction.
    
    The environment operates by:
    1. Setting up logging configuration for debugging and monitoring
    2. Initializing TORAX configuration and simulation state
    3. Managing discrete time steps with configurable time intervals
    4. Applying actions by updating TORAX configuration parameters
    5. Executing simulation steps and extracting observations
    6. Computing rewards and determining episode termination
    
    Attributes:
        observation_handler (Observation): Handles observation space and data extraction
        action_handler (ActionHandler): Manages action space and parameter updates
        config (ConfigLoader): TORAX configuration manager
        torax_app (ToraxApp): TORAX simulation wrapper
        state (dict): Current complete plasma state
        observation (dict): Current filtered observation
        T (float): Total simulation time [s]
        delta_t_a (float): Time interval between actions [s]
        current_time (float): Current simulation time [s]
        timestep (int): Current timestep counter
        terminated (bool): Episode termination flag
        truncated (bool): Episode truncation flag
    
    Abstract Methods:
        build_observation_variables(): Define observation space variables
        build_action_list(): Define available control actions
        reward(): Compute reward signal (optional override)
    """
    
    # Gymnasium metadata for rendering configuration
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        config: dict[str, Any],
        render_mode: str|None = None, 
        discretization_torax: str = "auto", 
        ratio_a_sim: int|None = None, 
        delta_t_a: float|None = None,
        log_level="warning",
        logfile=None,
        fig: viz.FigureProperties|None = None
    ) -> None:
        """
        Initialize the TORAX gymnasium environment.
        
        Args:
            config: TORAX configuration dictionary (required).
            render_mode: Rendering mode for visualization. Options: "human", "rgb_array", or None.
            discretization_torax: Time discretization method. Options:
                - "auto": Use explicit delta_t_a timing
                - "fixed": Use ratio of simulation timesteps
            ratio_a_sim: Ratio of action timesteps to simulation timesteps. 
                Required when discretization_torax="fixed".
            delta_t_a: Time interval between actions in seconds.
                Required when discretization_torax="auto".
            log_level: Logging level for environment operations. Options: "debug", "info", 
                "warning", "error", "critical". Default: "warning".
            logfile: Path to log file for writing log messages. If None, logs to console.
                
        Raises:
            ValueError: If required parameters are missing for chosen discretization method.
            TypeError: If discretization_torax is not "auto" or "fixed".
            
        Note:
            The environment must implement build_observation_variables() and 
            build_action_list() abstract methods to define the observation and action spaces.
            Logging is set up during initialization and applies to all environment operations.
        """
        setup_logging(getattr(logging, log_level.upper()), logfile)

        # Initialize observation and action handlers using abstract methods
        # These must be implemented by concrete subclasses
        self.observation_handler = self.build_observation_variables()
        self.action_handler = ActionHandler(self.build_action_list())
        
        # Initialize state tracking
        self.state: dict[str, Any]|None = None  # Plasma state
        self.observation: dict[str, Any]|None = None  # Observation

        # Load and validate TORAX configuration
        self.config: ConfigLoader = ConfigLoader(config)
        self.config.validate_discretization(discretization_torax)
    
        # Get total simulation time from configuration
        self.T: float = self.config.get_total_simulation_time()  # [seconds]
        
        # Configure time discretization based on chosen method
        if discretization_torax == "auto":
            # Use explicit action timestep timing
            if delta_t_a is None:
                raise ValueError("delta_t_a must be provided for auto discretization")
            self.delta_t_a: float = delta_t_a  # Time between actions [s]
        elif discretization_torax == "fixed":
            # Use ratio-based timing relative to simulation timesteps
            if ratio_a_sim is None:
                raise ValueError("ratio_a_sim must be provided for fixed discretization")
            delta_t_sim: float = self.config.get_simulation_timestep()  # TORAX internal timestep [s]
            self.delta_t_a: float = ratio_a_sim * delta_t_sim  # Action interval [s]
        else:
            raise TypeError(f"Invalid discretization method: {discretization_torax}. Use 'auto' or 'fixed'.")

        # Initialize time tracking
        self.current_time: float = 0.0  # Current simulation time [s]
        self.timestep: int = 0  # Current action timestep counter

        # Initialize TORAX simulation wrapper
        config_loader = ConfigLoader(config, self.action_handler)
        self.torax_app: ToraxApp = ToraxApp(config_loader, self.delta_t_a)

        # Update state/observation variables based on selected actions
        self.observation_handler.update_variables(self.action_handler.get_action_variables())

        # Build Gymnasium spaces
        self.action_space = self.action_handler.build_action_space()
        self.observation_handler.set_n_grid_points(self.config.get_n_grid_points())
        self.observation_space = self.observation_handler.build_observation_space()

        # Validate and set rendering mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.plotter = viz.ToraxStyleRealTimePlotter(fig, render_mode=self.render_mode)
        
        # Initialize rendering components (will be set up when needed)
        self.window = None
        self.clock = None

    def reset(
        self, 
        *, 
        seed: int|None = None, 
        options: dict[str, Any]|None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment to its initial state for a new episode.
        
        This method initializes a new simulation episode by:
        1. Resetting internal counters and flags
        2. Starting the TORAX simulation from initial conditions
        3. Extracting the initial observation state
        4. Optionally rendering the initial state
            
        Returns:
            Tuple containing:
            - observation (dict): Initial observation of plasma state
            - info (dict): Additional information (empty dict)
        """
        super().reset(seed=seed, options=options)
        
        # Reset episode flags
        self.terminated = False
        self.truncated = False
        self.timestep = 0
        self.current_time = 0.0
        
        # Initialize TORAX simulation
        self.torax_app.start()  # Set up initial simulation state
        torax_state = self.torax_app.get_state_data()  # Get initial plasma state
        
        # Extract initial observation
        self.state, self.observation = self.observation_handler.extract_state_observation(torax_state)

        # Render initial state if in human mode
        self.render()

        logger.debug(" environment reset complete.")

        return self.observation, {}

    def step(
        self, 
        action: NDArray[np.floating]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step with the given action.
        
        This method implements the core RL interaction by:
        1. Capturing the current state before action
        2. Applying the action to update TORAX configuration
        3. Running the simulation for one time interval
        4. Extracting the new observation state
        5. Computing the reward signal
        6. Checking for episode termination
        7. Updating time counters
        
        Args:
            action: Action array containing parameter values for all configured actions.
                
        Returns:
            Tuple containing:
            - observation (dict): New plasma state observation
            - reward (float): Reward signal for this step
            - terminated (bool): True if episode ended due to terminal condition
            - truncated (bool): True if episode ended due to time/step limits
            - info (dict): Additional step information
        """
        truncated = False
        info = {}
        
        # Capture current state before applying action
        state = self.torax_app.get_state_data()

        # Apply action by updating TORAX configuration parameters
        self.torax_app.update_config(action)
        
        # Execute simulation step
        success = self.torax_app.run()

        # Check if simulation completed successfully
        if not success:
            # Simulation failed - mark episode as terminated
            self.terminated = True

        # Extract new state and observation after simulation step
        next_torax_state = self.torax_app.get_state_data()
        next_state, observation = self.observation_handler.extract_state_observation(next_torax_state)
        self.state, self.observation = next_state, observation

        # Compute reward based on state transition
        reward = self.reward(state, next_state, action)
        
        # Update time tracking
        self.current_time += self.delta_t_a
        self.timestep += 1
        if self.current_time >= self.T:
            self.terminated = True

        # Render frame 
        self.render()
            
        return observation, reward, self.terminated, truncated, info 

    def reward(
        self, 
        state: dict[str, Any], 
        next_state: dict[str, Any], 
        action: NDArray[np.floating]
    ) -> float:
        """
        Compute the reward signal for a state transition.
        
        This method should be overridden by concrete subclasses to implement
        task-specific reward functions. The default implementation returns 0.0.
        
        Args:
            state: Previous plasma state before action was applied.
                Contains complete state with "profiles" and "scalars" dictionaries.
            next_state: New plasma state after action and simulation step.
                Same structure as state parameter.
            action: Action array that was applied to cause this transition.
                
        Returns:
            float: Reward value for this state transition.
            
        Example:
            >>> def reward(self, state, next_state, action):
            ...     # Reward based on proximity to target beta_N
            ...     target_beta = 2.0
            ...     current_beta = next_state["scalars"]["beta_N"]
            ...     return -abs(current_beta - target_beta)
        """
        return 0.0

    def close(self) -> None:
        """
        Clean up environment resources.
        
        This method properly closes the TORAX simulation and releases any
        rendering resources. Should be called when the environment is no
        longer needed.
        """
        # Close TORAX simulation
        self.torax_app.close()
        
        # Clean up rendering resources
        if self.window is not None:
            # TODO: Add pygame cleanup when rendering is implemented
            # pygame.display.quit()
            # pygame.quit()
            pass
    
    def render(self) -> None:
        """
        Render the current environment state following Gymnasium convention.
        
        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array", else None.
        
        Note:
            - For 'human' mode, this calls the plotter's update method for live visualization.
            - For 'rgb_array' mode, this calls the plotter's get_rgb_array() method (which must be implemented by the plotter).
            - Subclasses must provide a plotter compatible with these calls.
        """
        if self.render_mode == "human":
            if self.plotter is not None:
                self.plotter.update(self.state, self.current_time)
        else:
            if self.plotter is not None:
                self.plotter.update(self.state, self.current_time)

    def get_gif(self, filename: str) -> None:
        """
        Save the current plot as a GIF file.
        Args:
            filename: Path to save the GIF file. 
                If it does not end with ".gif", the suffix will be added.
        """
        if self.plotter is not None:
            # verify that the suffix is correct
            if not filename.endswith(".gif"):
                filename += ".gif"
            self.plotter.save_gif(filename)
        else:
            logger.warning("No plotter available to save GIF.")

    def _terminal_state(self) -> bool:
        """
        Check if the environment has reached a terminal state.
        
        This method can be overridden by subclasses to implement custom
        termination conditions based on plasma state, time limits, or
        other criteria.
        
        Returns:
            bool: True if episode should terminate, False otherwise.
            
        Note:
            Currently not implemented. Termination is handled by simulation
            failure detection in the step() method.
        """
        # TODO: Implement custom termination logic
        return False

    def _render_frame(self):
        """
        [DEPRECATED] Use render() instead. This method is kept for backward compatibility.
        """
        return self.render()


    # =============================================================================
    # Abstract Methods - Must be implemented by concrete subclasses
    # =============================================================================

    @abstractmethod
    def build_observation_variables(self) -> Observation:
        """
        Define the observation space variables for this environment.
        
        This method must be implemented by concrete subclasses to specify
        which TORAX variables should be included in the observation space.
        
        Returns:
            Observation: Configured observation handler that defines which
                plasma state variables are visible to the RL agent.
                
        Example:
            >>> def build_observation_variables(self):
            ...     return AllObservation(
            ...         exclude=["n_impurity", "Z_impurity"],
            ...         custom_bounds={
            ...             "T_e": (0.0, 50.0),  # Temperature range in keV
            ...             "T_i": (0.0, 50.0)
            ...         }
            ...     )
        """
        raise NotImplementedError
    
    @abstractmethod
    def build_action_list(self) -> list[Action]:
        """
        Define the available control actions for this environment.
        
        This method must be implemented by concrete subclasses to specify
        which plasma parameters can be controlled by the RL agent.
        
        Returns:
            List[Action]: List of Action instances representing controllable
                parameters with their bounds and TORAX configuration mappings.
                
        Example:
            >>> def build_action_list(self):
            ...     return [
            ...         IpAction(min=[0.5e6], max=[2.0e6]),      # Plasma current
            ...         EcrhAction(                               # ECRH heating
            ...             min=[0.0, 0.0, 0.0],                # [power, loc, width]
            ...             max=[10e6, 1.0, 0.5]
            ...         ),
            ...         NbiAction()                               # NBI with defaults
            ...     ]
        """
        raise NotImplementedError