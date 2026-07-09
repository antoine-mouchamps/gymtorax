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

from torax._src.orchestration.initial_state import (
    get_initial_state_and_post_processed_outputs_from_file,
)
from torax._src.output_tools.output import StateHistory
from torax._src.output_tools.post_processing import PostProcessedOutputs
from torax._src.state import SimError
from torax.experimental import (
    RuntimeParamsProvider,
    SimState,
    SimulationStepFn,
    get_initial_state_and_post_processed_outputs,
    make_step_fn,
)
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
        current_sim_state (SimState): Current simulation state
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
        self.current_sim_state: SimState | None = None
        self.current_sim_output: PostProcessedOutputs | None = None
        self.state: StateHistory | None = (
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
            - Physics models (transport, pedestal, sources, MHD, neoclassical)
            - Solver and step function for simulation advancement
            - Runtime parameters and geometry providers (owned by the step function)
            - Initial simulation state and outputs

        Called automatically by ``reset()`` if not already started.
        """
        # Build the step function from the config
        self.step_fn = make_step_fn(self.initial_config.config_torax)

        if (
            self.initial_config.config_torax.restart
            and self.initial_config.config_torax.restart.do_restart
        ):
            self.initial_sim_state, self.initial_sim_output = (
                get_initial_state_and_post_processed_outputs_from_file(
                    file_restart=self.initial_config.config_torax.restart,
                    step_fn=self.step_fn,
                )
            )
        else:
            self.initial_sim_state, self.initial_sim_output = (
                get_initial_state_and_post_processed_outputs(
                    t=self.initial_config.config_torax.numerics.t_initial,
                    step_fn=self.step_fn,
                )
            )

        # Create state history container with initial state
        state_history = StateHistory(
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

        # Create state history container with initial state
        state_history = StateHistory(
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

        # Propagate the new t_final to the step function's runtime params
        self.step_fn = SimulationStepFn(
            solver=self.step_fn.solver,
            time_step_calculator=self.step_fn.time_step_calculator,
            runtime_params_provider=RuntimeParamsProvider.from_config(
                self.config.config_torax
            ),
            geometry_provider=self.config.config_torax.geometry.build_provider,
        )

        logger.debug(" ToraxApp reset.")

    def run(self) -> tuple[bool, bool]:
        """Execute one simulation step from t_current to t_current + delta_t_a.

        This method advances the TORAX simulation by one action timestep, which may
        involve multiple internal TORAX timesteps. It handles:

        - Performance timing (if debug logging enabled)
        - TORAX execution via ``SimulationStepFn.jitted_fixed_time_step``
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

            # Advance the simulation by exactly one action timestep. This may
            # involve multiple internal physics timesteps. Only the state
            # at the end of the window is returned.
            output_state, post_processed_outputs = self.step_fn.jitted_fixed_time_step(
                self.delta_t_a,
                self.current_sim_state,
                self.current_sim_output,
            )

            # Check for errors between substeps
            sim_error = self.step_fn.check_for_errors(
                output_state, post_processed_outputs
            )
        except Exception as e:
            logger.error(
                f" an error occurred during the simulation run: {e}. The environment will reset"
            )
            return False, False

        # Check if TORAX simulation encountered internal errors
        if sim_error != SimError.NO_ERROR:
            logger.error("simulation terminated with an error.")
            sim_error.log_error()
            logger.error(" The environment will reset.")

            return False, False

        # Update current state to final state from simulation step
        self.current_sim_state = output_state
        self.current_sim_output = post_processed_outputs

        # Store results in history if history tracking is enabled
        if self.store_history is True:
            self.history_list.append([output_state, post_processed_outputs])

        # Update state history container with new results
        self.state = StateHistory(
            state_history=[self.current_sim_state],
            post_processed_outputs_history=[self.current_sim_output],
            sim_error=sim_error,
            torax_config=self.config.config_torax,
        )

        # Advance time by one action timestep
        self.t_current += self.delta_t_a

        # Check if we have reached the end of the episode
        if self.t_current >= self.t_final:
            logger.debug(" simulation run terminated successfully.")
            return True, True

        return True, False

    def update_config(self, action) -> None:
        """Update simulation configuration with new action parameters.

        This method applies new control parameters for the next simulation
        step: the step function's `RuntimeParamsProvider` is patched in place
        via ``update_provider_from_mapping`` (a jit-compatible leaf
        replacement, no pydantic re-validation). Recompilation is avoided by
        two combined guarantees: this mechanism cannot change the JAX pytree
        structure, and the action handler always emits fixed-shape
        (two-breakpoint) series, so leaf array shapes never change either
        (see ``Action.get_provider_updates``).

        Args:
            action: Action dictionary containing new parameter values.
                Must match the format expected by the ConfigLoader.

        Raises:
            ValueError: If action format is invalid or configuration update fails.
        """
        try:
            # Apply the action and collect the corresponding
            # runtime-parameter updates (two-breakpoint ramps)
            provider_updates = self.config.apply_action(
                action,
                self.t_current,  # Start time for this step
                self.delta_t_a,  # Action timestep duration
            )

            # t_final must track the end of the action window. The advance
            # length itself is set by the `dt` passed to
            # `jitted_fixed_time_step`, but torax compares t + dt against
            # numerics.t_final (`exact_t_final` handling), notably to allow a
            # sub-min_dt solver retry on the final, cropped substep of the
            # window.
            provider_updates["numerics.t_final"] = float(
                self.t_current + self.delta_t_a
            )

            # Patch the provider in place and rewrap the step function,
            # reusing the solver, time step calculator and geometry provider
            new_provider = (
                self.step_fn.runtime_params_provider.update_provider_from_mapping(
                    provider_updates
                )
            )
        except ValueError as e:
            raise ValueError(f"Error updating configuration: {e}")

        self.step_fn = SimulationStepFn(
            solver=self.step_fn.solver,
            time_step_calculator=self.step_fn.time_step_calculator,
            runtime_params_provider=new_provider,
            geometry_provider=self.step_fn.geometry_provider,
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

        state_history = StateHistory(
            state_history=state_history[start:end],
            post_processed_outputs_history=post_processed_outputs_history[start:end],
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax,
        )
        # TODO: Could we avoid this ? _to_xr takes as much time as the simulation itself
        dt = state_history.simulation_output_to_xr()

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

        state_history = StateHistory(
            state_history=state_history,
            post_processed_outputs_history=post_processed_outputs_history,
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax,
        )
        dt = state_history.simulation_output_to_xr()

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
