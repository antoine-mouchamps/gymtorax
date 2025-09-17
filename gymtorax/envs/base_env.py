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
    ...     def __init__(self, render_mode=None, **kwargs):
    ...         # Set environment-specific defaults
    ...         kwargs.setdefault("log_level", "info")
    ...         super().__init__(render_mode=render_mode, **kwargs)
    ...
    ...     def _define_observation_space(self):
    ...         return AllObservation(exclude=["n_impurity"])
    ...
    ...     def _define_action_space(self):
    ...         return [IpAction(), EcrhAction()]
    ...
    ...     def _get_torax_config(self):
    ...         return CONFIG
    ...
    ...     def _compute_reward(self, state, next_state, action):
    ...         # Custom reward logic
    ...         return -abs(next_state["scalars"]["beta_N"] - 2.0)
"""

import copy
import importlib
import logging
from abc import ABC, abstractmethod
from ctypes import ArgumentError
from typing import Any

import gymnasium as gym
import numpy as np
from numpy._typing._array_like import NDArray

import gymtorax.rendering.visualization as viz

from ..action_handler import Action, ActionHandler
from ..logger import setup_logging
from ..observation_handler import Observation
from ..torax_wrapper import ConfigLoader, ToraxApp, torax_plot_extensions

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

    Abstract Method:
        - _define_observation_space(): Define observation space variables
        - _define_action_space(): Define available control actions
        - _get_torax_config(): Define TORAX configuration parameters
        - _compute_reward(): Define reward signal (optional override)
    """

    # Gymnasium metadata for rendering configuration
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        log_level="warning",
        logfile=None,
        fig: viz.FigureProperties | None = None,  # TODO: REMOVE FROM HERE
        store_history=False,
    ) -> None:
        """Initialize the TORAX gymnasium environment.

        This method sets up the complete simulation environment including TORAX configuration,
        action/observation spaces, time discretization, and rendering components.

        Args:
            render_mode (str, optional): Rendering mode for visualization.
                Options: "human", "rgb_array", or None. Defaults to None.
            log_level (str, optional): Logging level for environment operations.
                Options: "debug", "info", "warning", "error", "critical".
                Defaults to "warning".
            logfile (str, optional): Path to log file for writing log messages.
                If None, logs to console. Defaults to None.
            fig (FigureProperties, optional): Figure properties for visualization
                configuration. Defines plot layout, variables to display, and styling
                options for real-time plotting and GIF creation. If None, default
                visualization settings will be used. Defaults to None.
            store_history (bool, optional): Whether to store simulation history
                for later saving. Set to True if you plan to use save_file() method.
                Defaults to False.

        Raises:
            ValueError: If required parameters are missing for chosen discretization method.
            TypeError: If discretization_torax is not "auto" or "fixed".
            KeyError: If required keys are missing from TORAX configuration.

        Note:
            Subclasses should use **kwargs to pass parameters to avoid explicit parameter
            listing and maintain flexibility as the base class evolves. Environment-specific
            defaults can be set using kwargs.setdefault() before calling super().__init__().

            The environment must implement the abstract methods _define_observation(),
            _define_action_space(), _get_torax_config(), and _compute_reward().
        """
        setup_logging(getattr(logging, log_level.upper()), logfile)

        try:
            config = copy.deepcopy(self._get_torax_config()["config"])
            discretization_torax = self._get_torax_config()["discretization"]
        except KeyError as e:
            raise KeyError(f"Missing key in TORAX config: {e}")

        # Initialize action handler using abstract method
        self.action_handler = ActionHandler(self._define_action_space())

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
            if self._get_torax_config()["delta_t_a"] is None:
                raise ValueError("delta_t_a must be provided for auto discretization")
            self.delta_t_a: float = self._get_torax_config()[
                "delta_t_a"
            ]  # Time between actions [s]
        elif discretization_torax == "fixed":
            # Use ratio-based timing relative to simulation timesteps
            if self._get_torax_config()["ratio_a_sim"] is None:
                raise ValueError(
                    "ratio_a_sim must be provided for fixed discretization"
                )
            delta_t_sim: float = (
                self.config.get_simulation_timestep()
            )  # TORAX internal timestep [s]
            self.delta_t_a: float = (
                self._get_torax_config()["ratio_a_sim"] * delta_t_sim
            )  # Action interval [s]
        else:
            raise TypeError(
                f"Invalid discretization method: {discretization_torax}. Use 'auto' or 'fixed'."
            )

        # Initialize time tracking
        self.current_time: float = 0.0  # Current simulation time [s]
        self.timestep: int = 0  # Current action timestep counter

        # Initialize TORAX simulation wrapper
        self.torax_app: ToraxApp = ToraxApp(self.config, self.delta_t_a, store_history)

        # Start simulator
        self.torax_app.start()

        # Initialize observation handler
        self.observation_handler = self._define_observation_space()

        # Set variables appearing in the actual simulation states
        self.observation_handler.set_state_variables(self.torax_app.get_state_data())

        # Set the variables appearing in the action, to be removed from the
        # state/observation
        self.observation_handler.set_action_variables(
            self.action_handler.get_action_variables()
        )

        # Build Gymnasium spaces
        # WARNING: At this stage, the observation space cannot be fully
        # determined. It is first set to the maximal possible space.
        self.action_space = self.action_handler.build_action_space()
        self.observation_space = self.observation_handler.build_observation_space()

        # Validate and set rendering mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if fig is not None:
            self.plotter = viz.ToraxStyleRealTimePlotter(
                fig, render_mode=self.render_mode
            )
        else:
            self.plotter = None

        self.store_history = store_history

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

        # Initialize last_action_dict for storing last actions
        self.last_action_dict = {}

        if self.plotter is not None:
            self.plotter.reset()

        # Initialize TORAX simulation
        self.torax_app.reset()  # Set up initial simulation state
        torax_state = self.torax_app.get_state_data()  # Get initial plasma state

        self.current_time = torax_state["/"]["time"][0].item()
        # Update last_action_dict with initial state
        self._update_last_action_dict(torax_state)

        # Extract initial observation
        self.state, self.observation = (
            self.observation_handler.extract_state_observation(torax_state)
        )

        if self.store_history:
            self.observation_history = [self.observation]
            self.actual_action_history = [
                self.torax_app.config.get_current_action_values()
            ]
            self.action_history = [self.torax_app.config.get_current_action_values()]

        if self.plotter is not None:
            self.plotter.update(
                current_state=self.state,
                action_input=self.last_action_dict,
                t=self.current_time,
            )
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
        actual_action_value = self.torax_app.config.get_current_action_values()

        # Check if action was clipped (due to bounds or ramp rate constraints)
        action_clipped = False
        clipped_actions = {}
        for key in action.keys():
            if not np.allclose(
                action[key], actual_action_value[key], rtol=1e-9, atol=1e-9
            ):
                action_clipped = True
                clipped_actions[key] = {
                    "requested": action[key].copy()
                    if hasattr(action[key], "copy")
                    else action[key],
                    "actual": actual_action_value[key].copy()
                    if hasattr(actual_action_value[key], "copy")
                    else actual_action_value[key],
                }

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

        # Update last_action_dict with new state
        self._update_last_action_dict(next_torax_state)

        next_state, observation = self.observation_handler.extract_state_observation(
            next_torax_state
        )
        self.state, self.observation = next_state, observation

        if self.store_history:
            self.observation_history.append(self.observation)
            self.actual_action_history.append(actual_action_value)
            self.action_history.append(action)

        # Compute reward based on state transition
        if not success or not self.observation_space.contains(self.observation):
            reward = -1000.0  # Large negative reward on failure
            self.terminated = True
            truncated = True
        else:
            reward = self._compute_reward(state, next_state, action)

        # Update time tracking
        self.current_time += self.delta_t_a
        self.timestep += 1
        if self.current_time > self.T:
            self.terminated = True

        if self.plotter is not None:
            self.plotter.update(
                current_state=self.state,
                action_input=self.last_action_dict,
                t=self.current_time,
            )

        # Add action clipping information to info dict
        info["action_clipped"] = action_clipped
        if action_clipped:
            info["clipped_actions"] = clipped_actions

        return observation, reward, self.terminated, truncated, info

    def close(self) -> None:
        """Clean up environment resources."""
        if self.plotter is not None:
            self.plotter.close()

    def render(self) -> None:
        """Render the current environment state following Gymnasium convention.

        Note:
            For 'human' mode, this calls the plotter's update method for live visualization.
        """
        if self.render_mode == "human" and self.plotter is not None:
            self.plotter.render_frame(t=self.current_time)

    def save_file(self, file_name):
        """Save the simulation output data to a file.

        This method saves the complete simulation history to a specified file.
        The simulation must have been initialized with store_history=True for
        this method to work properly.

        Args:
            file_name (str): The path and filename where the output should be saved.
                The file format is typically NetCDF (.nc extension).

        Raises:
            ArgumentError: If the environment was created without store_history=True.
            RuntimeError: If there was an error during the save operation.
        """
        try:
            self.torax_app.save_output_file(file_name)
        except RuntimeError as e:
            raise ArgumentError(
                "To save the output file, the store_history option must be set to True when creating the environment."
            ) from e

        logger.debug(f"Saved simulation history to {file_name}")

    def save_gif_torax(
        self,
        filename: str = "torax_evolution.gif",
        config_plot: str = "default",
        interval: int = 200,
        frame_skip: int = 2,
        beginning: int = 0,
        end: int = -1,
    ) -> None:
        """Generate and save an GIF of the simulation.

        This method loads a plotting configuration by name, extracts the simulation history (optionally
        selecting a time range), and generates an animated GIF visualizing the evolution of the simulation.
        The plot configuration must exist as a module in `torax.plotting.configs` and contain a `PLOT_CONFIG`
        attribute. The simulation must have been run with `store_history=True` for this to work.

        Args:
            config_plot (str): Name of the plot configuration to use (e.g., "default").
            filename (str): Output GIF filename.
            interval (int): Delay between frames in milliseconds.
            frame_skip (int): Save every Nth frame (default 2 = every other frame).
            beginning (int): Start time for the GIF (inclusive).
            end (int): End time for the GIF (inclusive, -1 for no upper limit).

        Raises:
            ImportError: If the plot configuration module cannot be found.
            AttributeError: If the module does not contain a PLOT_CONFIG attribute.
            RuntimeError: If the simulation was not run with state history enabled.
        """
        try:
            module = importlib.import_module(
                f"torax.plotting.configs.{config_plot}_plot_config"
            )
        except ImportError:
            logger.error(f"""Plot config: {config_plot} not found
                         in `torax.plotting.configs`""")
            return
        try:
            PLOT_CONFIG = getattr(module, "PLOT_CONFIG")
        except AttributeError:
            logger.error(f"""Plot config: {config_plot} does not have a PLOT_CONFIG attribute
                         in `torax.plotting.configs`""")
            return

        data_tree = self.torax_app.get_output_datatree(beginning, end)

        torax_plot_extensions.plot_run_to_gif(
            plot_config=PLOT_CONFIG,
            data_tree=data_tree,
            gif_filename=filename,
            n_frames=50,
            duration=interval,
            optimize=False,
            frame_skip=frame_skip,
        )

    def save_gif(
        self,
        filename: str = "torax_output.gif",
        interval: int = 200,
        frame_skip: int = 2,
    ) -> None:
        """Save the data as a GIF file.

        Args:
            filename: Path to save the GIF file.
                If it does not end with ".gif", the suffix will be added.
            interval: Delay between frames in ms.
            frame_skip: Save every Nth frame (default 2 = all frames).
                The last frame is always included.
        """
        if self.plotter is not None:
            # verify that the suffix is correct
            if not filename.endswith(".gif"):
                filename += ".gif"
            self.plotter.save_gif(filename, interval=interval, frame_skip=frame_skip)
        else:
            logger.warning("No plotter available to save GIF.")

    def _update_last_action_dict(self, state):
        """Update the last action dictionary with current state values.

        This internal method maintains a dictionary tracking the most recent values
        of action variables from the simulation state. It initializes the dictionary
        structure on first call and updates values on subsequent calls.

        The dictionary follows the structure: {'scalars': {...}, 'profiles': {...}}
        where keys correspond to action variable categories and values are the
        current state values for those variables.

        Args:
            state (dict): Current simulation state containing variable values
            organized by category (scalars, profiles, etc.)

        Note:
            This method is called during reset() and step() to keep track of
            action variable values for visualization and debugging purposes.
        """
        action_vars = self.action_handler.get_action_variables()
        # Initialize structure only if empty
        if not self.last_action_dict:
            self.last_action_dict = {"scalars": {}, "profiles": {}}
            for cat, var_list in action_vars.items():
                if cat not in self.last_action_dict:
                    self.last_action_dict[cat] = {}
                for var in var_list:
                    self.last_action_dict[cat][var] = None
        # Update values
        for cat, var_list in action_vars.items():
            for var in var_list:
                value = state.get(cat, {}).get(var, None)
                self.last_action_dict[cat][var] = value

    # =============================================================================
    # Abstract Methods - Must be implemented by concrete subclasses
    # =============================================================================

    @abstractmethod
    def _define_observation_space(self) -> Observation:
        """Define the observation space variables for this environment.

        This method must be implemented by concrete subclasses to specify
        which TORAX variables should be included in the observation space.

        Returns:
            Observation: Configured observation handler that defines which
            plasma state variables are visible to the RL agent.

        Example:
            >>> def _define_observation_space(self):
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
    def _define_action_space(self) -> list[Action]:
        """Define the available control actions for this environment.

        This method must be implemented by concrete subclasses to specify
        which plasma parameters can be controlled by the RL agent.

        Returns:
            List[Action]: List of Action instances representing controllable
            parameters with their bounds and TORAX configuration mappings.

        Example:
            >>> def _define_action_space(self):
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
    def _get_torax_config(self) -> dict[str, Any]:
        """Define the TORAX simulation configuration.

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


        Example:
            >>> def _get_torax_config(self):
            ...     return {
            ...         "config": TORAX_CONFIG,
            ...         "discretisation_torax": "auto",
            ...         "delta_t_a": 0.05,  # 50 ms between actions
            ...         # "ratio_a_sim": 10, # Only needed if using "fixed" discretization
            ...     }
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_reward(
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
            >>> def _compute_reward(self, state, next_state, action):
            ...     # Reward based on proximity to target beta_N
            ...     target_beta = 2.0
            ...     current_beta = next_state["scalars"]["beta_N"]
            ...     return -abs(current_beta - target_beta)
        """
        raise NotImplementedError
