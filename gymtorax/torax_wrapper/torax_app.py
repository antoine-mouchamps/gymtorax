from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop
from torax._src.orchestration import step_function
from torax._src.orchestration.sim_state import ToraxSimState
from torax._src.output_tools.post_processing import PostProcessedOutputs
from torax._src.output_tools import output
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import model_config
from torax._src.state import SimError
from torax._src import state
from xarray import DataTree
from numpy.typing import NDArray

from .config_loader import ConfigLoader
from . import torax_plot_extensions

import logging
import time

# Set up logger for this module
logger = logging.getLogger(__name__)

class ToraxApp:
    """Represents the Torax application. It initializes the application with a given configuration
    and provides methods to start the application, update the configuration, and run the simulation.
    
    Attributes:
        config: The configuration dictionary for the Torax application.
        delta_t_a: The time step for the actions.
        filename: The name of the file where the simulation state will be saved. The format is 
            'outputs/{filename}.nc'.
    
    Methods:
        __init__(config, delta_t_a, filename): Initializes the Torax application 
            with the provided configuration.
        start(): Initializes the Torax application with the provided configuration. 
        run(): Performs a "single" simulation "step" inside of TORAX (t_current -> t_current + delta_t_a).
        render(plot_configs, gif_name): Renders the simulation results using the provided plot configurations 
            and saves them to files.
    """
    def __init__(self, config_loader: ConfigLoader, delta_t_a: float, store_history: bool = False):
        self.store_history = store_history
        self.config: ConfigLoader = config_loader

        self.t_current = 0.0
        self.delta_t_a = delta_t_a
        self.t_final = self.config.get_total_simulation_time()
        self.config.set_total_simulation_time(self.delta_t_a) # End for the first action step
        
        self.is_start: bool = False     #Indicates if the application has been started
        
        self.geometry_provider = None
        self.static_runtime_params_slice = None
        self.dynamic_runtime_params_slice_provider = None
        self.step_fn = None
        self.post_processed_outputs = None  
        self.current_sim_state: ToraxSimState|None = None
        self.current_sim_output: PostProcessedOutputs|None = None
        self.state: output.StateHistory|None = None # history made up of a single state

        if self.store_history is True:
            self.history_list: list = []

        if logger.isEnabledFor(logging.DEBUG):
            self.last_run_time = None

    def start(self):
        """Initialize the Torax application with the provided configuration.
        This method sets up the transport model, pedestal model, geometry provider, source models,
        static runtime parameters slice, dynamic runtime parameters slice provider, solver, and step function.
        
        Returns:
            A tuple containing the transport model, pedestal model, geometry provider, source models,
            static runtime parameters slice, dynamic runtime parameters slice provider, step function,
            initial state, post-processed outputs, and a boolean indicating if the restart case is True.
        """

        transport_model = self.config.config_torax.transport.build_transport_model()
        pedestal_model = self.config.config_torax.pedestal.build_pedestal_model()

        self.geometry_provider = self.config.config_torax.geometry.build_provider
        
        source_models = source_models_lib.SourceModels(
            self.config.config_torax.sources, neoclassical=self.config.config_torax.neoclassical
        )

        self.static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(self.config.config_torax)
        )
        
        solver = self.config.config_torax.solver.build_solver(
            static_runtime_params_slice=self.static_runtime_params_slice,
            transport_model=transport_model,
            source_models=source_models,
            pedestal_model=pedestal_model,
        )

        mhd_models = self.config.config_torax.mhd.build_mhd_models(
            static_runtime_params_slice=self.static_runtime_params_slice,
            transport_model=transport_model,
            source_models=source_models,
            pedestal_model=pedestal_model,
        )

        self.step_fn = step_function.SimulationStepFn(
            solver=solver,
            time_step_calculator= self.config.config_torax.time_step_calculator.time_step_calculator,
            transport_model=transport_model,
            pedestal_model=pedestal_model,
            mhd_models=mhd_models,
        )

        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config.config_torax
            )
        )
        
        initial_sim_state, initial_sim_output = (
            initial_state_lib.get_initial_state_and_post_processed_outputs(
                t=self.config.config_torax.numerics.t_initial,
                static_runtime_params_slice=self.static_runtime_params_slice,
                dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                geometry_provider=self.geometry_provider,
                step_fn=self.step_fn,
            )
        )
        
        if self.store_history is True:
            self.history_list.append([initial_sim_state, initial_sim_output])

        self.current_sim_state = initial_sim_state
        self.current_sim_output = initial_sim_output
      
        state_history = output.StateHistory(
            state_history=[self.current_sim_state],
            post_processed_outputs_history=[self.current_sim_output],
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax
        )

        self.state = state_history

        # self.history_list.append((self.current_sim_state, self.current_sim_output))

        self.is_start = True

    def run(self)-> bool:
        """ Executes a single action step from `t_current` to `t_current + delta_t_a`.
        This action step may cover multiple simulation timesteps.
        This method runs the simulation loop for a single step, updating the simulation state and saving it
        to a NetCDF file if the simulation has reached the final time.
        Returns:
            A tuple containing the current state in xarray format and the history of the simulation.
        Raises:
            RuntimeError: If the application has not been started before running the simulation.
        """
        if self.is_start is False:
            raise RuntimeError("ToraxApp must be started before running the simulation.")
        
        if self.t_current >= self.t_final:
            logger.debug(" simulation run terminated successfully.")
            return True

        try: 
            if logger.isEnabledFor(logging.DEBUG):
                current_time = time.perf_counter()
                interval = current_time - self.last_run_time if self.last_run_time is not None else 0
            logger.debug(f" running simulation step at {self.t_current}/{self.t_final}s.")
            logger.debug(f" time since last run: {interval:.2f} seconds.")

            if logger.isEnabledFor(logging.DEBUG):
                self.last_run_time = current_time

            sim_states_list, post_processed_outputs_list, sim_error = run_loop.run_loop(
                static_runtime_params_slice=self.static_runtime_params_slice,
                dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                geometry_provider=self.geometry_provider,
                initial_state=self.current_sim_state,
                initial_post_processed_outputs=self.current_sim_output,
                restart_case=True,
                step_fn=self.step_fn,
                log_timestep_info=False,
                progress_bar=False,
            )
        except Exception as e:
            logger.error(f" an error occurred during the simulation run: {e}. The environment will reset")
            return False
        
        if(sim_error != state.SimError.NO_ERROR):
            logger.warning(f" simulation terminated with an error. The environment will reset")
            return False

        self.current_sim_state = sim_states_list[-1]
        self.current_sim_output = post_processed_outputs_list[-1]

        if self.store_history is True:
            self.history_list.append([sim_states_list[-1], post_processed_outputs_list[-1]])

        self.state = output.StateHistory(
            state_history=[self.current_sim_state],
            post_processed_outputs_history=[self.current_sim_output],
            sim_error=sim_error,
            torax_config=self.config.config_torax,
        )
        
        self.t_current += self.delta_t_a

        return True


    def update_config(self, action) -> None:
        """Update the configuration of the simulation based on the provided action.
        This method updates the configuration dictionary with new values for sources and profile conditions.
        It also prepares the restart file if necessary. 
        Args:
            action: The action to perform
        Returns:
            The updated configuration dictionary.
        """
        try:
            self.config.update_config(action,
                                    self.t_current,
                                    self.t_final,
                                    self.delta_t_a)
        except ValueError as e:
            raise ValueError(f"Error updating configuration: {e}")   
        
        self.geometry_provider = self.config.config_torax.geometry.build_provider
             
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config.config_torax
            )
        )

    # def render_gif(self, plot_configs: dict, gif_name: str)-> None:
    #     """Renders the simulation results using the provided plot configurations and saves them in a gif file.
    #     Args:
    #         plot_configs: A dictionary containing the plot configurations.
    #             Possible keys are 'default', 'simple', 'sources', 'global_params'.
    #             corresponding values are default_plot_config, simple_plot_config, sources_plot_config, global_params_plot_config.
    #         gif_name: The name of the GIF file to save the plots.
    #     """
    #     self.save_in_file()
    #     for plot_config in plot_configs:
    #         logger.debug(f" plotting with configuration: {plot_config}")
    #         torax_plot_extensions.plot_run_to_gif(
    #             plot_config=plot_configs[plot_config],
    #             outfile='WIP',
    #             frame_skip=5,
    #             gif_filename=f"tmp/{gif_name}_{plot_config}.gif", 
    #         )
    
    def save_output_file(self, file_name):
        """ Save in a .nc file the history """
        if(self.store_history is False):
            raise RuntimeError()

        state_history = [l[0] for l in self.history_list]
        post_processed_outputs_history = [l[1] for l in self.history_list]
        
        state_history = output.StateHistory(
            state_history=state_history,
            post_processed_outputs_history=post_processed_outputs_history,
            sim_error=SimError.NO_ERROR,
            torax_config=self.config.config_torax
        )
        dt = state_history.simulation_output_to_xr()
        try:
            dt.to_netcdf(file_name, engine="h5netcdf", mode="w")
        except Exception as e:
            raise ValueError(f"An error occurred while saving: {e}")

    def get_state_data(self):
        """_summary_
        """
        data = self.state.simulation_output_to_xr()
        
        return data