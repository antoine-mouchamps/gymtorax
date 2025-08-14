import torax
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop
from torax._src.orchestration import step_function
from torax._src.orchestration.sim_state import ToraxSimState
from torax._src.output_tools import output
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import model_config
from torax._src.state import SimError
from torax._src import state
from xarray import DataTree
import os
import tempfile

import torax_plot_extensions
from torax.plotting.configs.simple_plot_config import PLOT_CONFIG as simple_plot_config
from torax.plotting.configs.global_params_plot_config import PLOT_CONFIG as global_params_plot_config
from torax.plotting.configs.default_plot_config import PLOT_CONFIG as default_plot_config
from torax.plotting.configs.sources_plot_config import PLOT_CONFIG as sources_plot_config

from sources import Bounds, SourceBounds
from config_loader import ConfigLoader

"""To do

    - HANDLE pedestal model
    - Test sources
    - Create raise error if the application is not started/other issues
    
    - Prepare the bridge with gymnasium
    - Define the config file for users
    
    Suggestions: 
    - Create a source NBI
    - Try to create the nc file in the start function 
    
    Bugs:
    - Gérer la situation du 2ème exemples (De 0 à 50, cst à 0 mais de 0 à 100, 
        l'intervalle 0 à 50 augmente à cause de l'interpolation) 
"""

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
    def __init__(self, config: dict, delta_t_a: float):
        self.tmp_dir = None
        self.tmp_file_path = None
        self.t_current = 0.0
        self.delta_t_a = delta_t_a
        self.t_final = config['numerics']['t_final']
        config['numerics']['t_final'] = self.delta_t_a 
        
        self.config = ConfigLoader(config)
        self.config.validate()
        
        self.is_start: bool = False     #Indicates if the application has been started
        
        self.geometry_provider = None
        self.static_runtime_params_slice = None
        self.dynamic_runtime_params_slice_provider = None
        self.step_fn = None
        self.initial_state = None
        self.post_processed_outputs = None

        self.state_xr = None
        self.history = None  

    def start(self):
        """Initialize the Torax application with the provided configuration.
        This method sets up the transport model, pedestal model, geometry provider, source models,
        static runtime parameters slice, dynamic runtime parameters slice provider, solver, and step function.
        
        Returns:
            A tuple containing the transport model, pedestal model, geometry provider, source models,
            static runtime parameters slice, dynamic runtime parameters slice provider, step function,
            initial state, post-processed outputs, and a boolean indicating if the restart case is True.
        """
        
        # Create an empty output.nc file for compatibility with torax
        # restart mechanics
        fd, self.tmp_file_path = tempfile.mkstemp(suffix=".nc", prefix="gymtorax_", dir=None)
        os.close(fd)
        
        # Check if the file can be accessed
        if self.tmp_file_path is None or not os.path.exists(self.tmp_file_path):
            raise FileNotFoundError("Output file not found.")
        
        self.config.setup_for_simulation(self.tmp_file_path)

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
        
        self.initial_state, self.post_processed_outputs = (
            initial_state_lib.get_initial_state_and_post_processed_outputs(
                t=self.config.config_torax.numerics.t_initial,
                static_runtime_params_slice=self.static_runtime_params_slice,
                dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                geometry_provider=self.geometry_provider,
                step_fn=self.step_fn,
            )
        )
              
        state_history = output.StateHistory(
            state_history=[self.initial_state],
            post_processed_outputs_history=[self.post_processed_outputs],
            sim_error=SimError(0),
            torax_config=self.config.config_torax
        )
        
        self.state_xr = state_history.simulation_output_to_xr(file_restart=None)

        self.state_xr.to_netcdf(self.tmp_file_path, engine="h5netcdf", mode="w")

        self.is_start = True

    def update_config(self, action) -> None:
        """Update the configuration of the simulation based on the provided action.
        This method updates the configuration dictionary with new values for sources and profile conditions.
        It also prepares the restart file if necessary. 
        Args:
            action: A dictionary containing the new configuration values for sources and profile conditions.
        Returns:
            The updated configuration dictionary.
        """
        self.config.update_config(action,
                                  self.t_current,
                                  self.t_final,
                                  self.delta_t_a)
                
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config.config_torax
            )
        )


    def run(self)-> tuple:
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
            print("Simulation has reached the final time, no further steps will be executed.")
            print("By default, the simulation saves the current state to 'outputs/{self.filename}.nc'.")
            self.state.to_netcdf(f'outputs/{self.filename}.nc', engine="h5netcdf", mode="w")
            return (
                self.state_xr,
                self.history,
            )

        state_history, post_processed_outputs_history, sim_error = run_loop.run_loop(
            static_runtime_params_slice=self.static_runtime_params_slice,
            dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
            geometry_provider=self.geometry_provider,
            initial_state=self.initial_state,
            initial_post_processed_outputs=self.post_processed_outputs,
            restart_case=True,
            step_fn=self.step_fn,
            log_timestep_info=False,
            progress_bar=False,
        )
        state_history = output.StateHistory(
            state_history=state_history,
            post_processed_outputs_history=post_processed_outputs_history,
            sim_error=sim_error,
            torax_config=self.config.config_torax,
        )
        self.t_current += self.delta_t_a
        self.state_xr = state_history.simulation_output_to_xr(self.config.config_torax.restart)
        self.history = state_history
        
        # If the simulation has reached the final time, save the full history
        if self.t_current >= self.t_final:
            self.state_xr.to_netcdf(self.tmp_file_path, engine="h5netcdf", mode="w")
            print(f"Simulation state updated in {self.tmp_file_path}.")

        return True

    def render_gif(self, plot_configs: dict, gif_name: str)-> None:
        """Renders the simulation results using the provided plot configurations and saves them in a gif file.
        Args:
            plot_configs: A dictionary containing the plot configurations.
                Possible keys are 'default', 'simple', 'sources', 'global_params'.
                corresponding values are simple_plot_config, sources_plot_config, global_params_plot_config.
            gif_name: The name of the GIF file to save the plots.
        Raises:
            RuntimeError: If the state is None or if the application has not been started before rendering.
        """
        os.makedirs('plots', exist_ok=True)
        if self.state_xr is None:
            raise RuntimeError("State is None. Please run the simulation first.")
        #self.state_xr.to_netcdf(self.tmp_file_path, engine="h5netcdf", mode="w")
        
        for plot_config in plot_configs:
            print(f"Plotting with configuration: {plot_config}")
            torax_plot_extensions.plot_run_to_gif(
                plot_config=plot_configs[plot_config],
                outfile=self.tmp_file_path,
                gif_filename=f"plots/{gif_name}_{plot_config}.gif", 
                frame_skip=5,
            )
        
    def close(self):
        """Close TORAX ? DELETE OUTPUT FILE(s)"""
        # Clean up everything
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        self.temp_file_path = None


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


if __name__ == "__main__":
    import config_file_test
    
    config_test = config_file_test.CONFIG
    
    torax_env = ToraxApp(config_test, delta_t_a=50)
    
    if False:
        torax_env.start()
        torax_env.run()
        torax_env.render(plot_configs={'default': default_plot_config}, name='torax_iter_long1')
        torax_env.update_config({'profile_conditions': {'Ip': {60: 6.0e6}}})
        torax_env.run()
        torax_env.render(plot_configs={'default': default_plot_config}, name='torax_iter_long2')
        torax_env.update_config({'profile_conditions': {'Ip': {120: 10.0e6}}})
        torax_env.run()
        torax_env.render(plot_configs={'default': default_plot_config}, name='torax_iter_long3')
        
    if True:
        torax_env.start()
        torax_env.run()
        torax_env.render_gif(plot_configs={'sources': sources_plot_config}, gif_name='torax_iter_long_ecrh1')
        torax_env.update_config({'sources': {'ecrh': {'gaussian_location': 0.4, 'gaussian_width': 0.15, 
                                                            'P_total': {torax_env.t_current: 10e6, torax_env.t_current + torax_env.delta_t_a/2: 10e6}}}})
        torax_env.run()
        torax_env.render_gif(plot_configs={'sources': sources_plot_config}, gif_name='torax_iter_long_ecrh2')
        torax_env.update_config(None)
        torax_env.run()
        torax_env.render_gif(plot_configs={'sources': sources_plot_config}, gif_name='torax_iter_long_ecrh3')
        torax_env.close()