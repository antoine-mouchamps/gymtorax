"""TORAX Base Environment Module.

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
    ...     def define_observation(self):
    ...         return AllObservation(exclude=["n_impurity"])
    ...
    ...     def define_actions(self):
    ...         return [IpAction(), EcrhAction()]
    ...
    ...     def define_reward(self, state, next_state, action):
    ...         # Custom reward logic
    ...         return -abs(next_state["scalars"]["beta_N"] - 2.0)
"""

import logging
from abc import ABC, abstractmethod
from ctypes import ArgumentError
from typing import Any

import gymnasium as gym
import numpy as np
from numpy._typing._array_like import NDArray

from ..action_handler import Action, ActionHandler
from ..logger import setup_logging
from ..observation_handler import Observation
from ..torax_wrapper import ConfigLoader, ToraxApp

# Set up logger for this module
logger = logging.getLogger(__name__)


class BaseEnv(gym.Env, ABC):
    """Abstract base class for TORAX plasma simulation environments.

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
        define_observation(): Define observation space variables
        define_actions(): Define available control actions
        define_reward(): Define reward signal (optional override)
    """

    # Gymnasium metadata for rendering configuration
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        log_level="warning",
        logfile=None,
        store_state_history=False,
    ) -> None:
        """Initialize the TORAX gymnasium environment.

        Args:
            render_mode: Rendering mode for visualization. Options: "human", "rgb_array", or None.
            log_level: Logging level for environment operations. Options: "debug", "info",
                "warning", "error", "critical". Default: "warning".
            logfile: Path to log file for writing log messages. If None, logs to console.

        Raises:
            ValueError: If required parameters are missing for chosen discretization method.
            TypeError: If discretization_torax is not "auto" or "fixed".

        Note:
            The environment must implement define_observation() and
            define_actions() abstract methods to define the observation and action spaces.
            Logging is set up during initialization and applies to all environment operations.
        """
        setup_logging(getattr(logging, log_level.upper()), logfile)

        try:
            config = self.get_torax_config()["config"]
            discretization_torax = self.get_torax_config()["discretization"]
        except KeyError as e:
            raise KeyError(f"Missing key in TORAX config: {e}")

        # Initialize action handler using abstract method
        self.action_handler = ActionHandler(self.define_actions())

        # Initialize state tracking
        self.state: dict[str, Any] | None = None  # Plasma state
        self.observation: dict[str, Any] | None = None  # Observation

        # Load and validate TORAX configuration
        self.config: ConfigLoader = ConfigLoader(config, self.action_handler)
        self.config.validate_discretization(discretization_torax)

        # Get total simulation time from configuration
        self.T: float = self.config.get_total_simulation_time()  # [seconds]

        # Configure time discretization based on chosen method
        if discretization_torax == "auto":
            # Use explicit action timestep timing
            if self.get_torax_config()["delta_t_a"] is None:
                raise ValueError("delta_t_a must be provided for auto discretization")
            self.delta_t_a: float = self.get_torax_config()[
                "delta_t_a"
            ]  # Time between actions [s]
        elif discretization_torax == "fixed":
            # Use ratio-based timing relative to simulation timesteps
            if self.get_torax_config()["ratio_a_sim"] is None:
                raise ValueError(
                    "ratio_a_sim must be provided for fixed discretization"
                )
            delta_t_sim: float = (
                self.config.get_simulation_timestep()
            )  # TORAX internal timestep [s]
            self.delta_t_a: float = (
                self.get_torax_config()["ratio_a_sim"] * delta_t_sim
            )  # Action interval [s]
        else:
            raise TypeError(
                f"Invalid discretization method: {discretization_torax}. Use 'auto' or 'fixed'."
            )

        # Initialize time tracking
        self.current_time: float = 0.0  # Current simulation time [s]
        self.timestep: int = 0  # Current action timestep counter

        # Initialize TORAX simulation wrapper
        self.store_state_history = store_state_history
        self.torax_app: ToraxApp = ToraxApp(
            self.config, self.delta_t_a, store_state_history
        )

        # Start simulator
        self.torax_app.start()

        # Initialize observation handler
        self.observation_handler = self.define_observation()

        # Set variables appearing in the actual simulation states
        self.observation_handler.set_state_variables(self.torax_app.get_state_data())

        # Set the variables appearing in the action, to be removed from the
        # state/observation
        self.observation_handler.set_action_variables(
            self.action_handler.get_action_variables()
        )

        self.observation_handler.set_n_grid_points(self.config.get_n_grid_points())

        # Build Gymnasium spaces
        # WARNING: At this stage, the observation space cannot be fully
        # determined. It is first set to the maximal possible space.
        self.action_space = self.action_handler.build_action_space()
        self.observation_space = self.observation_handler.build_observation_space()

        # Validate and set rendering mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize rendering components (will be set up when needed)
        self.window = None
        self.clock = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment to its initial state for a new episode.

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
        self.torax_app.reset()  # Set up initial simulation state
        torax_state = self.torax_app.get_state_data()  # Get initial plasma state

        # Extract initial observation
        self.state, self.observation = (
            self.observation_handler.extract_state_observation(torax_state)
        )

        # Render initial state if in human mode
        if self.render_mode == "human":
            self._render_frame()

        logger.debug(" environment reset complete.")

        return self.observation, {}

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with the given action.

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
        success, terminated_simulation = self.torax_app.run()

        # Simulation failed - mark episode as terminated
        if not success:
            self.terminated = True

        # Simulation is done, episode is terminated
        if terminated_simulation:
            self.terminated = True

        # Extract new state and observation after simulation step
        next_torax_state = self.torax_app.get_state_data()
        next_state, observation = self.observation_handler.extract_state_observation(
            next_torax_state
        )
        self.state, self.observation = next_state, observation

        # Compute reward based on state transition
        reward = self.define_reward(state, next_state, action)

        # Update time tracking
        self.current_time += self.delta_t_a
        self.timestep += 1

        # If simulation reached final time, terminate the environment
        if self.current_time > self.T:
            self.terminated = True

        # Render frame if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self.terminated, truncated, info

    def define_reward(
        self,
        state: dict[str, Any],
        next_state: dict[str, Any],
        action: NDArray[np.floating],
    ) -> float:
        """Define the reward signal for a state transition.

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
            >>> def define_reward(self, state, next_state, action):
            ...     # Reward based on proximity to target beta_N
            ...     target_beta = 2.0
            ...     current_beta = next_state["scalars"]["beta_N"]
            ...     return -abs(current_beta - target_beta)
        """
        return 0.0

    def close(self) -> None:
        """Clean up environment resources.

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

    def render(self):
        """Render the current environment state.

        Returns:
            NDArray or None: RGB array if render_mode is "rgb_array", else None.

        Note:
            Currently not fully implemented. Returns None for all modes.
        """
        if self.render_mode == "human":
            pass
        if self.render_mode == "rgb_array" or self.render_mode == None:
            pass
        return None

    def save_file(self, file_name):
        """"""
        try:
            self.torax_app.save_output_file(file_name)
        except RuntimeError as e:
            raise ArgumentError(
                "To save the output file, the store_history option must be set to True when creating the environment."
            ) from e

        logger.debug(f"Saved simulation history to {file_name}")

    # =============================================================================
    # Abstract Methods - Must be implemented by concrete subclasses
    # =============================================================================

    @abstractmethod
    def define_observation(self) -> Observation:
        """Define the observation space variables for this environment.

        This method must be implemented by concrete subclasses to specify
        which TORAX variables should be included in the observation space.

        Returns:
            Observation: Configured observation handler that defines which
                plasma state variables are visible to the RL agent.

        Example:
            >>> def define_observation(self):
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
    def define_actions(self) -> list[Action]:
        """Define the available control actions for this environment.

        This method must be implemented by concrete subclasses to specify
        which plasma parameters can be controlled by the RL agent.

        Returns:
            List[Action]: List of Action instances representing controllable
                parameters with their bounds and TORAX configuration mappings.

        Example:
            >>> def define_actions(self):
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

    @abstractmethod
    def get_torax_config(self) -> dict[str, Any]:
        """Get the TORAX simulation configuration.

        This abstract method must be implemented by concrete subclasses
        which provides the necessary parameters for the TORAX simulation,
        including its core configuration, the time discretization method,
        the control time step, and the ratio between simulation and control time steps.

        Returns:
            Dict[str, Any]: A dictionary containing the TORAX configuration.
                The dictionary must have the following keys:
                - "config" (dict): A dictionary of TORAX configuration parameters.
                - "discretisation_torax" (str): The time discretization method.
                    Options are "auto" (uses 'delta_t_a') or "fixed" (uses 'ratio_a_sim').
                - "ratio_a_sim" (int, optional): The ratio of action timesteps to
                    simulation timesteps. Required if 'discretisation_torax' is "fixed".
                - "delta_t_a" (float, optional): The time interval between actions
                    in seconds. Required if 'discretisation_torax' is "auto".
        """
        raise NotImplementedError
