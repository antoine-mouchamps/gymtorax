from abc import ABC, abstractmethod
from dataclasses import dataclass
import profile

import numpy as np

import numpy as np
from torax._src.output_tools import output
import xarray as xr

class StateVariable:
    # Class-level attributes to be overridden by subclasses
    default_min: list = None  
    default_max: list = None
    
    def __init__(self, name: str, vals: list=None, min_vals: float=None, max_vals: float=None):
        self.dimension = len(vals)
        # Use provided values or defaults
        self._min = [min_vals]*self.dimension or [self.default_min]*self.dimension
        self._max = [max_vals]*self.dimension or [self.default_max]*self.dimension

        # Validate bounds dimensions
        if self._min is not None and len(self._min) != self.dimension:
            raise ValueError(f"Invalid min dimension {len(self._min)}, must be {self.dimension}")
        if self._max is not None and len(self._max) != self.dimension:
            raise ValueError(f"Invalid max dimension {len(self._max)}, must be {self.dimension}")
        
        # Set defaults if None
        if self._min is None:
            self._min = [0.0] * self.dimension
        if self._max is None:
            self._max = [np.inf] * self.dimension


    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.value}, min={self.min[0]}, max={self.max[0]})"


class State:
    def __init__(self, data_tree: xr.DataTree):
        time = data_tree[output.TIME].to_numpy()
        
        def get_optional_data(ds, key, grid_type):
            if grid_type.lower() not in ['cell', 'face']:
                raise ValueError(
                    f'grid_type for {key} must be either "cell" or "face", got'
                    f' {grid_type}'
                )
            if key in ds:
                return ds[key].to_numpy()
            else:
                return (
                    np.zeros((len(time), len(ds[output.RHO_CELL_NORM])))
                    if grid_type == 'cell'
                    else np.zeros((len(time), len(ds[output.RHO_FACE_NORM].to_numpy())))
                )

        def _transform_data(ds: xr.Dataset):
            """Transforms data in-place to the desired units."""
            # TODO(b/414755419)
            ds = ds.copy()

            transformations = {
                output.J_TOTAL: 1e6,  # A/m^2 to MA/m^2
                output.J_OHMIC: 1e6,  # A/m^2 to MA/m^2
                output.J_BOOTSTRAP: 1e6,  # A/m^2 to MA/m^2
                output.J_EXTERNAL: 1e6,  # A/m^2 to MA/m^2
                'j_generic_current': 1e6,  # A/m^2 to MA/m^2
                output.I_BOOTSTRAP: 1e6,  # A to MA
                output.IP_PROFILE: 1e6,  # A to MA
                'j_ecrh': 1e6,  # A/m^2 to MA/m^2
                'p_icrh_i': 1e6,  # W/m^3 to MW/m^3
                'p_icrh_e': 1e6,  # W/m^3 to MW/m^3
                'p_generic_heat_i': 1e6,  # W/m^3 to MW/m^3
                'p_generic_heat_e': 1e6,  # W/m^3 to MW/m^3
                'p_ecrh_e': 1e6,  # W/m^3 to MW/m^3
                'p_alpha_i': 1e6,  # W/m^3 to MW/m^3
                'p_alpha_e': 1e6,  # W/m^3 to MW/m^3
                'p_ohmic_e': 1e6,  # W/m^3 to MW/m^3
                'p_bremsstrahlung_e': 1e6,  # W/m^3 to MW/m^3
                'p_cyclotron_radiation_e': 1e6,  # W/m^3 to MW/m^3
                'p_impurity_radiation_e': 1e6,  # W/m^3 to MW/m^3
                'ei_exchange': 1e6,  # W/m^3 to MW/m^3
                'P_ohmic_e': 1e6,  # W to MW
                'P_aux_total': 1e6,  # W to MW
                'P_alpha_total': 1e6,  # W to MW
                'P_bremsstrahlung_e': 1e6,  # W to MW
                'P_cyclotron_e': 1e6,  # W to MW
                'P_ecrh': 1e6,  # W to MW
                'P_radiation_e': 1e6,  # W to MW
                'I_ecrh': 1e6,  # A to MA
                'I_aux_generic': 1e6,  # A to MA
                'W_thermal_total': 1e6,  # J to MJ
                output.N_E: 1e20,  # m^-3 to 10^{20} m^-3
                output.N_I: 1e20,  # m^-3 to 10^{20} m^-3
                output.N_IMPURITY: 1e20,  # m^-3 to 10^{20} m^-3
            }

            for var_name, scale in transformations.items():
                if var_name in ds:
                    ds[var_name] /= scale

            return ds

        data_tree = xr.map_over_datasets(_transform_data, data_tree)
        profiles_dataset = data_tree.children[output.PROFILES].dataset
        scalars_dataset = data_tree.children[output.SCALARS].dataset
        dataset = data_tree.dataset
        
        self.T_i = profiles_dataset["T_i"].to_numpy()[0]
        self.T_e = profiles_dataset["T_e"].to_numpy()[0]
        self.psi = profiles_dataset["psi"].to_numpy()[0]
        self.n_e = profiles_dataset["n_e"].to_numpy()[0]
        self.n_i = profiles_dataset["n_i"].to_numpy()[0]
        self.n_impurity = profiles_dataset["n_impurity"].to_numpy()[0]
        self.q = profiles_dataset["q"].to_numpy()[0]
        self.magnetic_shear = profiles_dataset["magnetic_shear"].to_numpy()[0]
        self.v_loop = profiles_dataset["v_loop"].to_numpy()[0]
        self.Z_i = profiles_dataset["Z_i"].to_numpy()[0]
        self.Z_impurity = profiles_dataset["Z_impurity"].to_numpy()[0]
        self.Z_eff = profiles_dataset["Z_eff"].to_numpy()[0]
        self.sigma_parallel = profiles_dataset["sigma_parallel"].to_numpy()[0]
        self.j_total = profiles_dataset["j_total"].to_numpy()[0]
        self.Ip = profiles_dataset["Ip_profile"].to_numpy()[0]
        self.Q_fusion=scalars_dataset['Q_fusion'].to_numpy()[0]  # pylint: disable=invalid-name
        # self.j_ohmic = profiles_dataset[output.J_OHMIC].to_numpy()[0]
        # self.j_bootstrap = profiles_dataset[output.J_BOOTSTRAP].to_numpy()[0]
        # self.j_external = profiles_dataset[output.J_EXTERNAL].to_numpy()[0]
        # self.j_ecrh=get_optional_data(profiles_dataset, 'j_ecrh', 'cell')[0]
        # self.j_generic_current=get_optional_data(
        #     profiles_dataset, 'j_generic_current', 'cell'
        # )[0]
        # self.chi_turb_i = profiles_dataset[output.CHI_TURB_I].to_numpy()[0]
        # self.chi_turb_e = profiles_dataset[output.CHI_TURB_E].to_numpy()[0]
        # self.D_turb_e = profiles_dataset[output.D_TURB_E].to_numpy()[0]
        # self.V_turb_e = profiles_dataset[output.V_TURB_E].to_numpy()[0]
        # self.rho_norm=dataset[output.RHO_NORM].to_numpy()[0]
        # self.rho_cell_norm=dataset[output.RHO_CELL_NORM].to_numpy()[0]
        # self.rho_face_norm=dataset[output.RHO_FACE_NORM].to_numpy()[0]
        # self.p_icrh_i=get_optional_data(profiles_dataset, 'p_icrh_i', 'cell')[0]
        # self.p_icrh_e=get_optional_data(profiles_dataset, 'p_icrh_e', 'cell')[0]
        # self.p_generic_heat_i=get_optional_data(
        #     profiles_dataset, 'p_generic_heat_i', 'cell'
        # )[0]
        # self.p_generic_heat_e=get_optional_data(
        #     profiles_dataset, 'p_generic_heat_e', 'cell'
        # )[0]
        # self.p_ecrh_e=get_optional_data(profiles_dataset, 'p_ecrh_e', 'cell')[0]
        # self.p_alpha_i=get_optional_data(profiles_dataset, 'p_alpha_i', 'cell')[0]
        # self.p_alpha_e=get_optional_data(profiles_dataset, 'p_alpha_e', 'cell')[0]
        # self.p_ohmic_e=get_optional_data(profiles_dataset, 'p_ohmic_e', 'cell')[0]
        # self.p_bremsstrahlung_e=get_optional_data(
        #     profiles_dataset, 'p_bremsstrahlung_e', 'cell'
        # )[0]
        # self.p_cyclotron_radiation_e=get_optional_data(
        #     profiles_dataset, 'p_cyclotron_radiation_e', 'cell'
        # )[0]
        # self.p_impurity_radiation_e=get_optional_data(
        #     profiles_dataset, 'p_impurity_radiation_e', 'cell'
        # )[0]
        # self.ei_exchange = profiles_dataset[
        #     'ei_exchange'
        # ].to_numpy()[0]  # ion heating/sink
        # self.s_gas_puff=get_optional_data(profiles_dataset, 's_gas_puff', 'cell')[0]
        # self.s_generic_particle=get_optional_data(
        #     profiles_dataset, 's_generic_particle', 'cell'
        # )[0]
        # self.s_pellet=get_optional_data(profiles_dataset, 's_pellet', 'cell')[0]
        # self.Ip_profile = profiles_dataset[output.IP_PROFILE].to_numpy()[:, -1][0]
        # self.I_bootstrap=scalars_dataset[output.I_BOOTSTRAP].to_numpy()[0]
        # self.I_aux_generic=scalars_dataset['I_aux_generic'].to_numpy()[0]
        # self.I_ecrh=scalars_dataset['I_ecrh'].to_numpy()[0]
        # self.P_ohmic_e=scalars_dataset['P_ohmic_e'].to_numpy()[0]
        # self.P_auxiliary=scalars_dataset['P_aux_total'].to_numpy()[0]
        # self.P_alpha_total=scalars_dataset['P_alpha_total'].to_numpy()[0]
        # self.P_sink=scalars_dataset['P_bremsstrahlung_e'].to_numpy()[0] \
        #     + scalars_dataset['P_radiation_e'].to_numpy()[0] \
        #     + scalars_dataset['P_cyclotron_e'].to_numpy()[0]
        # self.P_bremsstrahlung_e=scalars_dataset['P_bremsstrahlung_e'].to_numpy()[0]
        # self.P_radiation_e=scalars_dataset['P_radiation_e'].to_numpy()[0]
        # self.P_cyclotron_e=scalars_dataset['P_cyclotron_e'].to_numpy()[0]
        # self.T_e_volume_avg=scalars_dataset['T_e_volume_avg'].to_numpy()[0]
        # self.T_i_volume_avg=scalars_dataset['T_i_volume_avg'].to_numpy()[0]
        # self.n_e_volume_avg=scalars_dataset['n_e_volume_avg'].to_numpy()[0]
        # self.n_i_volume_avg=scalars_dataset['n_i_volume_avg'].to_numpy()[0]
        # self.W_thermal_total=scalars_dataset['W_thermal_total'].to_numpy()[0]
        # self.q95=scalars_dataset['q95'].to_numpy()[0]
        # self.t=time[0]