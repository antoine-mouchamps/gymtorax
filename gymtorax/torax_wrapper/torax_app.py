"""High-level application interface for running TORAX plasma simulations.

This module provides the `ToraxApp` class, which wraps the TORAX simulator
into a Pythonic interface suitable for reinforcement learning and episodic
simulation workflows. It manages the simulation lifecycle, configuration updates,
state tracking, and output handling.

This abstraction allows Gymnasium-style environments and control algorithms
to interact with TORAX without dealing with its low-level orchestration details.
"""

import copy
import logging
import time

from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop, step_function
from torax._src.orchestration.sim_state import ToraxSimState
from torax._src.output_tools import output
from torax._src.output_tools.post_processing import PostProcessedOutputs
from torax._src.sources import source_models as source_models_lib
from torax._src.state import SimError
from xarray import DataTree

from .config_loader import ConfigLoader

# Set up logger for this module
logger = logging.getLogger(__name__)


class ToraxApp:
    """TORAX simulation application wrapper.

    This class provides a high-level interface for running TORAX plasma simulations
    in an episodic manner, suitable for reinforcement learning environments. It manages
    the simulation lifecycle, state tracking, and configuration updates.

    The application follows a start/reset -> run -> update cycle:
        1. Initialize with configuration and action timestep
        2. Call ``reset()`` to prepare for a new episode
        3. Call ``run()`` repeatedly to advance the simulation
        4. Call ``update_config()`` between runs to update action parameters

    Attributes:
        config (ConfigLoader): Current configuration loader instance
        initial_config (ConfigLoader): Original configuration for resetting
        delta_t_a (float): Action timestep - simulation duration per ``run()``
        store_history (bool): Whether to store complete simulation history
        current_sim_state (ToraxSimState): Current simulation state
        current_sim_output (PostProcessedOutputs): Current post-processed outputs
        state (StateHistory): Current state history (single timestep)
        history_list (list): Complete history list (if ``store_history=True``)
        is_started (bool): Whether the application has been initialized
        t_current (float): Current simulation time
        t_final (float): Final simulation time for current episode
        last_run_time (float): Timestamp of last ``run()`` call (for performance monitoring)
    """

    def __init__(
        self, config_loader: ConfigLoader, delta_t_a: float, store_history: bool = False
    ):
        """Initialize ToraxApp with configuration and simulation parameters.

        Args:
            config_loader: `ConfigLoader` instance containing TORAX configuration
            delta_t_a: Action timestep in seconds. Each call to ``run()`` advances
                simulation by this amount
            store_history: If ``True``, stores complete simulation history for later
                analysis. If ``False``, only keeps current state (more memory efficient)

        Note:
            The application must be ``reset()`` before first use. The constructor only
            sets up instance variables and enables performance monitoring if debug
            logging is enabled.
        """
        # Store configuration and simulation parameters
        self.store_history = store_history
        self.initial_config: ConfigLoader = config_loader
        self.delta_t_a = delta_t_a

        # Initialize state containers (will be populated by reset())
        self.current_sim_state: ToraxSimState | None = None
        self.current_sim_output: PostProcessedOutputs | None = None
        self.state: output.StateHistory | None = (
            None  # Current state history (single timestep)
        )

        # Optional full history storage for analysis/debugging
        if self.store_history is True:
            self.history_list: list = []

        # Performance monitoring (only if debug logging enabled)
        if logger.isEnabledFor(logging.DEBUG):
            self.last_run_time = None

        # Track initialization state
        self.is_started: bool = False

    def start(self):
        """Initialize TORAX simulation components.

        This method sets up all the TORAX simulation infrastructure:
            - Transport and pedestal models
            - Geometry provider and source models
            - Static and dynamic runtime parameters
            - Solver and MHD models
            - Step function for simulation advancement
            - Initial simulation state and outputs

        Called automatically by ``reset()`` if not already started.
        """
        # Build physics models for plasma simulation
        transport_model = (
            self.initial_config.config_torax.transport.build_transport_model()
        )
        pedestal_model = (
            self.initial_config.config_torax.pedestal.build_pedestal_model()
        )

        # Geometry provider defines the spatial grid and magnetic geometry
        self.geometry_provider = (
            self.initial_config.config_torax.geometry.build_provider
        )

        # Source models handle heating, current drive, and other plasma sources
        source_models = source_models_lib.SourceModels(
            self.initial_config.config_torax.sources,
            neoclassical=self.initial_config.config_torax.neoclassical,
        )

        # Static runtime parameters (do not change during simulation)
        self.static_runtime_params_slice = (
            build_runtime_params.build_static_params_from_config(
                self.initial_config.config_torax
            )
        )

        # Build the main physics solver (handles transport equations)
        solver = self.initial_config.config_torax.solver.build_solver(
            static_runtime_params_slice=self.static_runtime_params_slice,
            transport_model=transport_model,
            source_models=source_models,
            pedestal_model=pedestal_model,
        )

        # MHD models for magnetohydrodynamic instabilities and limits
        mhd_models = self.initial_config.config_torax.mhd.build_mhd_models(
            static_runtime_params_slice=self.static_runtime_params_slice,
            transport_model=transport_model,
            source_models=source_models,
            pedestal_model=pedestal_model,
        )

        # Main simulation step function - advances physics by one timestep
        self.step_fn = step_function.SimulationStepFn(
            solver=solver,
            time_step_calculator=self.initial_config.config_torax.time_step_calculator.time_step_calculator,
            transport_model=transport_model,
            pedestal_model=pedestal_model,
            mhd_models=mhd_models,
        )

        # Dynamic runtime parameters (can change during simulation via actions)
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.initial_config.config_torax
            )
        )

        if (
            self.initial_config.config_torax.restart
            and self.initial_config.config_torax.restart.do_restart
        ):
            self.initial_sim_state, self.initial_sim_output = (
                initial_state_lib.get_initial_state_and_post_processed_outputs_from_file(
                    t_initial=self.initial_config.config_torax.numerics.t_initial,
                    file_restart=self.initial_config.config_torax.restart,
                    static_runtime_params_slice=self.static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.geometry_provider,
                    step_fn=self.step_fn,
                )
            )
        else:
            self.initial_sim_state, self.initial_sim_output = (
                initial_state_lib.get_initial_state_and_post_processed_outputs(
                    t=self.initial_config.config_torax.numerics.t_initial,
                    static_runtime_params_slice=self.static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.geometry_provider,
                    step_fn=self.step_fn,
                )
            )

        # Create state history container with initial state
        state_history = output.StateHistory(
            state_history=[self.initial_sim_state],
            post_processed_outputs_history=[self.initial_sim_output],
            sim_error=SimError.NO_ERROR,
            torax_config=self.initial_config.config_torax,
        )

        # Update state and configuration references
        self.state = state_history

        # Mark as initialized
        self.is_started = True

        logger.debug(" ToraxApp started.")

    def reset(self):
        """Reset the simulation to initial conditions for a new episode.

        This method prepares the application for a new simulation episode by:
            - Initializing TORAX components if not already started
            - Resetting simulation state to initial conditions
            - Creating fresh state history
            - Setting up time tracking (t_current=0, t_final from config)
            - Configuring first action step duration
        """
        # Initialize TORAX physics models if not already done
        if self.is_started is False:
            self.start()

        # Store initial state in history if history tracking is enabled
        if self.store_history is True:
            self.history_list: list = []
            self.history_list.append((self.initial_sim_state, self.initial_sim_output))

        # Reset current simulation state to initial conditions
        self.current_sim_state = self.initial_sim_state
        self.current_sim_output = self.initial_sim_output

        # Rebuild geometry provider with initial configuration
        # This handles geometry changes (e.g., current profile modifications)
        self.geometry_provider = (
            self.initial_config.config_torax.geometry.build_provider
        )

        # Rebuild dynamic runtime parameters provider with initial configuration
        # This ensures time-dependent parameters reflect the updated config
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.initial_config.config_torax
            )
        )

        # Create state history container with initial state
        state_history = output.StateHistory(
            state_history=[self.current_sim_state],
            post_processed_outputs_history=[self.current_sim_output],
            sim_error=SimError.NO_ERROR,
            torax_config=self.initial_config.config_torax,
        )

        # Update state and configuration references
        self.state = state_history
        self.config = copy.deepcopy(self.initial_config)

        # Reset time tracking to beginning of episode
        if (
            self.initial_config.config_torax.restart
            and self.initial_config.config_torax.restart.do_restart
        ):
            self.t_current = self.config.get_initial_simulation_time(restart=True)
        else:
            self.t_current = self.config.get_initial_simulation_time()

        self.t_final = self.config.get_total_simulation_time()

        # Set simulation end time to first action timestep
        self.config.set_total_simulation_time(
            self.delta_t_a
        )  # End for the first action step

        logger.debug(" ToraxApp reset.")

    def run(self) -> tuple[bool, bool]:
        """Execute one simulation step from t_current to t_current + delta_t_a.

        This method advances the TORAX simulation by one action timestep, which may
        involve multiple internal TORAX timesteps. It handles:

        - Performance timing (if debug logging enabled)
        - TORAX run_loop execution with current configuration
        - State and output management
        - Error handling and recovery
        - Time progression tracking

        Returns:
            tuple[bool, bool]:
                - success (bool): True if simulation step completed successfully,
                    False if an error occurred or simulation reached final time.
                - done (bool): True if whole simulation is done.

        Raises:
            RuntimeError: If reset() has not been called before running.

        Note:
            - Call update_config() between runs to modify simulation parameters
            - Returns True when `t_current >= t_final` (episode complete)
            - Performance timing logged at DEBUG level shows interval since last run
            - Errors during simulation return False (environment should reset)
        """
        # Ensure simulation has been initialized
        if self.is_started is False:
            raise RuntimeError(
                "ToraxApp must be started before running the simulation."
            )

        try:
            # Performance monitoring - track time between runs
            if logger.isEnabledFor(logging.DEBUG):
                current_time = time.perf_counter()
                interval = (
                    current_time - self.last_run_time
                    if self.last_run_time is not None
                    else 0
                )
                logger.debug(
                    f" running simulation step at {self.t_current}/{self.t_final}s."
                )
                logger.debug(f" time since last run: {interval:.2f} seconds.")

            # Update timing reference for next run
            if logger.isEnabledFor(logging.DEBUG):
                self.last_run_time = current_time

            # Execute TORAX simulation loop for one action timestep
            # This may involve multiple internal physics timesteps
            sim_states_list, post_processed_outputs_list, sim_error = run_loop.run_loop(
                static_runtime_params_slice=self.static_runtime_params_slice,
                dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                geometry_provider=self.geometry_provider,
                initial_state=self.current_sim_state,
                initial_post_processed_outputs=self.current_sim_output,
                restart_case=True,  # Continue from current state
                step_fn=self.step_fn,
                log_timestep_info=False,  # Suppress internal TORAX logging
                progress_bar=False,  # No progress bar for individual steps
            )
        except Exception as e:
            logger.error(
                f" an error occurred during the simulation run: {e}. The environment will reset"
            )
            return False, False

        # Check if TORAX simulation encountered internal errors
        if sim_error != state.SimError.NO_ERROR:
            logger.error("simulation terminated with an error.")
            sim_error.log_error()
            logger.error(" The environment will reset.")

            return False, False

        # Update current state to final state from simulation step
        self.current_sim_state = sim_states_list[-1]
        self.current_sim_output = post_processed_outputs_list[-1]

        # Store results in history if history tracking is enabled
        if self.store_history is True:
            self.history_list.append(
                [sim_states_list[-1], post_processed_outputs_list[-1]]
            )

        # Update state history container with new results
        self.state = output.StateHistory(
            state_history=[self.current_sim_state],
            post_processed_outputs_history=[self.current_sim_output],
            sim_error=sim_error,
            torax_config=self.config.config_torax,
        )

        # Advance time by one action timestep
        self.t_current += self.delta_t_a

        # Check if we have reached the end of the episode
        if self.t_current > self.t_final:
            logger.debug(" simulation run terminated successfully.")
            return True, True

        return True, False

    def update_config(self, action) -> None:
        """Update simulation configuration with new action parameters.

        This method applies new control parameters to the TORAX configuration
        for the next simulation step.

        Args:
            action: Action dictionary containing new parameter values.
                Must match the format expected by the ConfigLoader.

        Raises:
            ValueError: If action format is invalid or configuration update fails.
        """
        try:
            # Apply action parameters to configuration with time constraints
            self.config.update_config(
                action,
                self.t_current,  # Start time for this step
                self.delta_t_a,
            )  # Action timestep duration
        except ValueError as e:
            raise ValueError(f"Error updating configuration: {e}")

        # Rebuild geometry provider with updated configuration
        # This handles geometry changes (e.g., current profile modifications)
        self.geometry_provider = self.config.config_torax.geometry.build_provider

        # Rebuild dynamic runtime parameters provider with new configuration
        # This ensures time-dependent parameters reflect the updated config
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config.config_torax
            )
        )

    def get_output_datatree(self, start: int = 0, end: int = -1) -> DataTree:
        """Return the full simulation history as an xarray DataTree.

        This method reconstructs the complete trajectory of the simulation,
        including all state and post-processed output snapshots, as an xarray
        DataTree suitable for analysis and visualization. If `beginning` and
        `end` are specified, only data between those time values (inclusive)
        will be selected for all datasets in the DataTree that have a 'time'
        coordinate. Requires that the ToraxApp was initialized with
        `store_history=True` so that the full history is available.

        Args:
            start (int or float): Start time for selection.
                Defaults to ``0``.
            end (int or float): End time for selection. Defaults to
                ``-1`` (no upper limit).

        Returns:
            xarray.DataTree: The complete simulation history as an xarray DataTree,
                with all timesteps and outputs, or only the selected time range
                if specified.

        Raises:
            RuntimeError: If ``store_history`` was not enabled and thus no
                history is available.
        """
        if self.store_history is False:
            raise RuntimeError()

        state_history = [output[0] for output in self.history_list]
        post_processed_outputs_history = [output[1] for output in self.history_list]

        state_history = output.StateHistory(
            state_history=state_history[start:end],
            post_processed_outputs_history=post_processed_outputs_history[start:end],
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax,
        )
        dt = state_history.simulation_output_to_xr(self.config.config_torax.restart)

        return dt

    def save_output_file(self, file_name):
        """Save complete simulation history to NetCDF file.

        This method saves the full simulation trajectory to a NetCDF file suitable
        for analysis and visualization. Requires ``store_history=True`` in constructor.

        Args:
            file_name (str): Output file path with .nc extension

        Raises:
            RuntimeError: If ``store_history=False`` (no history to save)
            ValueError: If file writing fails
        """
        if self.store_history is False:
            raise RuntimeError()

        state_history = [output[0] for output in self.history_list]
        post_processed_outputs_history = [output[1] for output in self.history_list]

        state_history = output.StateHistory(
            state_history=state_history,
            post_processed_outputs_history=post_processed_outputs_history,
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax,
        )
        dt = state_history.simulation_output_to_xr(self.config.config_torax.restart)

        try:
            dt.to_netcdf(file_name, engine="h5netcdf", mode="w")
        except Exception as e:
            raise ValueError(f"An error occurred while saving: {e}")

    def get_state_data(self):
        """Get current simulation state as xarray DataTree.

        This method returns the current simulation state in xarray format,
        suitable for observation extraction and analysis.

        Returns:
            xarray.DataTree: Current simulation state.

        Raises:
            RuntimeError: If simulation state has not been computed yet.

        Note:
            - Returns single-timestep state (current moment)
            - For full history, use ``save_output_file()`` with ``store_history=True``
        """
        if self.state is None:
            raise RuntimeError("Simulation state has not been computed yet.")

        data = self.state.simulation_output_to_xr()

        return data
