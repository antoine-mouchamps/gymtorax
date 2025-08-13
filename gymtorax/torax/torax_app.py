import torax
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import model_config
import xarray as xr
import torax_plot_extensions
from torax.plotting.configs.simple_plot_config import PLOT_CONFIG as simple_plot_config
from torax.plotting.configs.global_params_plot_config import PLOT_CONFIG as global_params_plot_config
from torax.plotting.configs.default_plot_config import PLOT_CONFIG as default_plot_config
from torax.plotting.configs.sources_plot_config import PLOT_CONFIG as sources_plot_config
from matplotlib import pyplot as plt
import matplotlib
import os
from dataclasses import dataclass

@dataclass
class Bounds:
    min: float
    max: float

@dataclass
class SourceBounds:
    total: Bounds
    loc: Bounds
    width: Bounds
    
def expand_sources(ES_k_bounds:list[SourceBounds]) -> list[Bounds]:
    """Transform the list of SourceBounds (tuples of Bounds) into the corresponding
    list of bounds (deleting the SourceBounds wrapper)

    Returns
    -------
    list
        The expanded action space for the sources.
    """
    return [bounds for source in ES_k_bounds for bounds in (source.total, source.loc, source.width)]
    

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
        ratio_ta_tsim: The ratio of delta_t_a to tsim.
        filename: The name of the file where the simulation state will be saved. The format is 
            'outputs/{filename}.nc'.
    
    Methods:
        __init__(config, delta_t_a, ratio_ta_tsim, filename): Initializes the Torax application 
            with the provided configuration.
        start(): Initializes the Torax application with the provided configuration. 
        update_config(action): Updates the configuration of the simulation based on the provided action.
        run(): Performs a "single" simulation "step" inside of TORAX (t_current -> t_current + delta_t_a).
        render(plot_configs, gif_name): Renders the simulation results using the provided plot configurations 
            and saves them to files.
    """
    def __init__(self, config: dict, delta_t_a: float, ratio_ta_tsim: int, filename: str):
        # Difficulty: Default values are possible but they are not provided in config.
        #It is hard to check without raising an error. It is possible to check inside
        #config_dict but it is more annoying to modify inside it.
        
        self.filename = filename
        
        self.delta_t_a = delta_t_a
        self.ratio_ta_tsim = ratio_ta_tsim
        config['numerics']['fixed_dt'] = delta_t_a / ratio_ta_tsim
        self.t_final = config['numerics']['t_final']
        
        if 't_initial' not in config['numerics']:
            config['numerics']['t_final'] = self.delta_t_a   
            self.t_current = 0.0
        else :
            config['numerics']['t_final'] = config['numerics']['t_initial'] + self.delta_t_a 
            self.t_current = config['numerics']['t_initial']
            
        config['numerics']['evolve_current'] = True
        config['numerics']['evolve_density'] = True
        config['time_step_calculator']['calculator_type'] = 'fixed'
        
        self.config = config
        self.config_dict = torax.ToraxConfig.from_dict(config)
        
        self.is_start: bool = False     #Indicates if the application has been started
        
        self.transport_model = None
        self.pedestal_model = None
        self.geometry_provider = None
        self.source_models = None
        self.static_runtime_params_slice = None
        self.dynamic_runtime_params_slice_provider = None
        self.step_fn = None
        self.initial_state = None
        self.post_processed_outputs = None
        
        self.restart_case = False   #Needs to be set to True AFTER the first run to generate the restart file
        
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
        self.transport_model = self.config_dict.transport.build_transport_model()
        self.pedestal_model = self.config_dict.pedestal.build_pedestal_model()

        self.geometry_provider = self.config_dict.geometry.build_provider
        self.source_models = source_models_lib.SourceModels(
            self.config_dict.sources, neoclassical=self.config_dict.neoclassical
        )

        self.static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(self.config_dict)
        )

        self.solver = self.config_dict.solver.build_solver(
        static_runtime_params_slice=self.static_runtime_params_slice,
        transport_model=self.transport_model,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        )

        self.mhd_models = self.config_dict.mhd.build_mhd_models(
            static_runtime_params_slice=self.static_runtime_params_slice,
            transport_model=self.transport_model,
            source_models=self.source_models,
            pedestal_model=self.pedestal_model,
        )

        self.step_fn = step_function.SimulationStepFn(
            solver=self.solver,
            time_step_calculator= self.config_dict.time_step_calculator.time_step_calculator,
            transport_model=self.transport_model,
            pedestal_model=self.pedestal_model,
            mhd_models=self.mhd_models,
        )

        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config_dict
            )
        )
        # Manage the restart section in the config
        if self.config_dict.restart and self.config_dict.restart.do_restart:
            self.initial_state, self.post_processed_outputs = (
                initial_state_lib.get_initial_state_and_post_processed_outputs_from_file(
                    t_initial=self.config_dict.numerics.t_initial,
                    file_restart=self.config_dict.restart,
                    static_runtime_params_slice=self.static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.geometry_provider,
                    step_fn=self.step_fn,
                )
            )
            restart_case = True
        else:
            self.initial_state, self.post_processed_outputs = (
                initial_state_lib.get_initial_state_and_post_processed_outputs(
                    t=self.config_dict.numerics.t_initial,
                    static_runtime_params_slice=self.static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.geometry_provider,
                    step_fn=self.step_fn,
                )
            )
            restart_case = False
            
        self.is_start = True
        

    def update_config(self, action: dict) -> dict:
        """Update the configuration of the simulation based on the provided action.
        This method updates the configuration dictionary with new values for sources and profile conditions.
        It also prepares the restart file if necessary. 
        Args:
            action: A dictionary containing the new configuration values for sources and profile conditions.
        Returns:
            The updated configuration dictionary.
        Raises:
            RuntimeError: If the application has not been started before updating the configuration.
        """
        if not self.is_start:
            raise RuntimeError("ToraxApp must be started before updating the config.")
        
        self.config['numerics']['t_initial'] = self.t_current
        self.config['numerics']['t_final'] = self.t_current + self.delta_t_a
        if self.config['numerics']['t_final'] > self.t_final:
            self.config['numerics']['t_final'] = self.t_final
            
        if self.t_current >= self.t_final:
            print("Simulation has reached the final time, no further steps will be executed.")
            return self.config
        
        if not action:
            print("No action provided, returning current config.")
        else:
            keys = action.keys()
            # Unlike other variables, 'sources' and 'profile_conditions' require manual merging.
            # TORAX does not update these profiles automatically; it keeps the last state for the entire simulation.
            # This code merges the existing data (before the current time) with the new data.
            if 'sources' in keys: 
                for source_name, new_source_profile in action['sources'].items():
                    old_source_profile = self.config['sources'].get(source_name, {})
                    merged_source_profile = {}

                    for param, new_val in new_source_profile.items():
                        old_val = old_source_profile.get(param)

                        # Si la valeur est un profil temporel (dict)
                        if isinstance(new_val, dict):
                            merged_val = {}

                            # Conserver ancien < t_current
                            if isinstance(old_val, dict):
                                for t, val in old_val.items():
                                    if float(t) < self.t_current:
                                        merged_val[t] = val

                            # Ajouter nouveau ≥ t_current
                            for t, val in new_val.items():
                                if float(t) >= self.t_current:
                                    merged_val[t] = val

                            merged_source_profile[param] = merged_val

                        else:
                            # Valeur scalaire → remplacer directement
                            merged_source_profile[param] = new_val

                    self.config['sources'][source_name] = merged_source_profile

            if 'profile_conditions' in keys:
                if 'Ip' in action['profile_conditions']:
                    new_ip_profile = action['profile_conditions']['Ip']
                    old_ip_profile = self.config['profile_conditions'].get('Ip', {})
                    print(old_ip_profile)
                    merged_ip_profile = {}

                    for t, val in old_ip_profile.items():
                        if float(t) < self.t_current:
                            merged_ip_profile[t] = val
                    for t, val in new_ip_profile.items():
                        if float(t) >= self.t_current:
                            merged_ip_profile[t] = val

                    self.config['profile_conditions']['Ip'] = merged_ip_profile

                if 'V_loop' in action['profile_conditions']:
                    new_vloop_profile = action['profile_conditions']['V_loop']
                    old_vloop_profile = self.config['profile_conditions'].get('V_loop', {})
                    merged_vloop_profile = {}

                    for t, val in old_vloop_profile.items():
                        if float(t) < self.t_current:
                            merged_vloop_profile[t] = val
                    for t, val in new_vloop_profile.items():
                        if float(t) >= self.t_current:
                            merged_vloop_profile[t] = val

                    self.config['profile_conditions']['V_loop'] = merged_vloop_profile
                    
        #Prepare the restart file
        restart_path = f".\\outputs\\{self.filename}.nc"
        if 'restart' not in self.config:
            self.config['restart'] = {
                'do_restart': os.path.isfile(restart_path),
                'filename': restart_path,
                'time': self.t_current,
                'stitch': True,
            }
        else:
            self.config['restart']['filename'] = restart_path
            self.config['restart']['time'] = self.t_current
            # Active le restart seulement si le fichier existe
            self.config['restart']['do_restart'] = os.path.isfile(restart_path)
            self.config['restart']['stitch'] = True

        
        self.config_dict = torax.ToraxConfig.from_dict(self.config)
        self.dynamic_runtime_params_slice_provider = (
            build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                self.config_dict
            )
        )
        
        return self.config

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
            restart_case=self.restart_case,
            step_fn=self.step_fn,
            log_timestep_info=False,
            progress_bar=False,
        )
        state_history = output.StateHistory(
            state_history=state_history,
            post_processed_outputs_history=post_processed_outputs_history,
            sim_error=sim_error,
            torax_config=self.config_dict,
        )
        self.t_current += self.delta_t_a
        self.state_xr = state_history.simulation_output_to_xr(self.config_dict.restart)
        self.history = state_history
        
        # Create the outputs directory and the output file if they don't exist
        if self.restart_case == False:
            self.restart_case = True
            os.makedirs('outputs', exist_ok=True)
            self.state_xr.to_netcdf(f'outputs/{self.filename}.nc',engine="h5netcdf", mode="w")
            print(f"Simulation completed and state saved to 'outputs/{self.filename}.nc'.")
        
        # If the simulation has reached the final time, save the full history
        if self.t_current >= self.t_final:
            self.state_xr.to_netcdf(f'outputs/{self.filename}.nc', engine="h5netcdf", mode="w")
            print(f"Simulation state updated in 'outputs/{self.filename}.nc'.")
            
        return (
                self.state_xr,
                self.history,
        )
        
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
        self.state_xr.to_netcdf(f'outputs/{self.filename}.nc', engine="h5netcdf", mode="w")
        
        for plot_config in plot_configs:
            print(f"Plotting with configuration: {plot_config}")
            torax_plot_extensions.plot_run_to_gif(
                plot_config=plot_configs[plot_config],
                outfile=f"outputs/{self.filename}.nc",
                gif_filename=f"plots/{gif_name}_{plot_config}.gif", 
                frame_skip=2,
            )
        
    def close(self):
        """Close TORAX ? DELETE OUTPUT FILE(s)"""
        pass
        
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
    
    torax_env = ToraxApp(config_test, delta_t_a=50, ratio_ta_tsim=50, filename='torax_iter_long')
    
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
