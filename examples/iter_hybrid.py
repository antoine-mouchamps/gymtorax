import numpy as np

import gymtorax.action_handler as ah
import gymtorax.observation_handler as oh

# fmt: off
import gymtorax.rendering.visualization as viz
from gymtorax import BaseAgent, BaseEnv
from gymtorax.rendering.plots import main_prop_fig

"""Config for ITER hybrid scenario based parameters with nonlinear solver.

ITER hybrid scenario based (roughly) on van Mulders Nucl. Fusion 2021.
With Newton-Raphson solver and adaptive timestep (backtracking)
"""

_NBI_W_TO_MA = 1/16e6 # rough estimate of NBI heating power to current drive
W_to_Ne_ratio = 0

# No NBI during rampup. Rampup all NBI power between 99-100 seconds
nbi_times = np.array([0, 99, 100])
nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA

# Gaussian prescription of "NBI" deposition profiles and fractional deposition
r_nbi = 0.25
w_nbi = 0.25
el_heat_fraction = 0.66

# No ECCD power for this config (but kept here for future flexibility)
eccd_power = {0: 0, 99: 0, 100: 20.0e6}


CONFIG = {
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},  # (bundled isotope average)
        'impurity': {'Ne': 1 - W_to_Ne_ratio, 'W': W_to_Ne_ratio},
        'Z_eff': {0.0: {0.0: 2.0, 1.0: 2.0}},  # sets impurity densities
    },
    'profile_conditions': {
        'Ip': {0: 3e6, 100: 12.5e6},  # total plasma current in MA
        'T_i': {0.0: {0.0: 6.0, 1.0: 0.2}}, # T_i initial condition
        'T_i_right_bc': 0.2, # T_i boundary condition
        'T_e': {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_e initial condition
        'T_e_right_bc': 0.2,  # T_e boundary condition
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': {0: 0.35, 100: 0.35}, # n_e boundary condition
        # set initial condition density according to Greenwald fraction.
        'nbar': 0.85, # line average density for initial condition
        'n_e': {0: {0.0: 1.3, 1.0: 1.0}},  # Initial electron density profile
        'normalize_n_e_to_nbar': True, # normalize initial n_e to nbar
        'n_e_nbar_is_fGW': True, # nbar is in units for greenwald fraction
        'initial_psi_from_j': True, # initial psi from current formula
        'initial_j_is_total_current': True, # only ohmic current on init
        'current_profile_nu': 2, # exponent in initial current formula
    },
    'numerics': {
        't_final': 150,  # length of simulation time in seconds
        'fixed_dt': 1, # fixed timestep
        'evolve_ion_heat': True, # solve ion heat equation
        'evolve_electron_heat': True, # solve electron heat equation
        'evolve_current': True, # solve current equation
        'evolve_density': True, # solve density equation
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,  # major radius (R) in meters
        'a_minor': 2.0,  # minor radius (a) in meters
        'B_0': 5.3,  # Toroidal magnetic field on axis [T]
    },
    'sources': {
        # Current sources (for psi equation)
        'ecrh': { # ECRH/ECCD (with Lin-Liu)
           'gaussian_width': 0.05,
           'gaussian_location': 0.35,
           'P_total': eccd_power,
           },
        'generic_heat': { # Proxy for NBI heat source
            'gaussian_location': r_nbi, # Gaussian location in normalized coordinates
            'gaussian_width': w_nbi, # Gaussian width in normalized coordinates
            'P_total': (nbi_times, nbi_powers), # Total heating power
            # electron heating fraction r
            'electron_heat_fraction': el_heat_fraction,
        },
        'generic_current': { # Proxy for NBI current source
            'use_absolute_current': True, # I_generic is total external current
            'gaussian_width': w_nbi,
            'gaussian_location': r_nbi,
            'I_generic': (nbi_times, nbi_cd),
        },
        'fusion': {}, # fusion power
        'ei_exchange': {}, # equipartition
        'ohmic': {}, # ohmic power
        'cyclotron_radiation': {}, # cyclotron radiation
        'impurity_radiation': { # impurity radiation + bremsstrahlung
            'model_name': 'mavrin_fit',
            'radiation_multiplier': 0.0,
        },
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        # use internal boundary condition model (for H-mode and L-mode)
        'set_pedestal': True,
        'T_i_ped': {0: 0.5, 100: 0.5, 105: 3.0},
        'T_e_ped': {0: 0.5, 100: 0.5, 105: 3.0},
        'n_e_ped_is_fGW': True,
        'n_e_ped': 0.85, # pedestal top n_e in units of fGW
        'rho_norm_ped_top': 0.95,  # set ped top location in normalized radius
    },
    'transport': {
        'model_name': 'qlknn',  # Using QLKNN_7_11 default
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        'apply_inner_patch': True,
        'D_e_inner': 0.15,
        'V_e_inner': 0.0,
        'chi_i_inner': 0.3,
        'chi_e_inner': 0.3,
        'rho_inner': 0.1,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.95,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        'chi_min': 0.05,  # minimum chi
        'chi_max': 100,  # maximum chi (can be helpful for stability)
        'D_e_min': 0.05,  # minimum electron diffusivity
        'D_e_max': 50,  # maximum electron diffusivity
        'V_e_min': -10,  # minimum electron convection
        'V_e_max': 10,  # minimum electron convection
        'smoothing_width': 0.1,
        'DV_effective': True,
        'include_ITG': True,  # to toggle ITG modes on or off
        'include_TEM': True,  # to toggle TEM modes on or off
        'include_ETG': True,  # to toggle ETG modes on or off
        'avoid_big_negative_s': False,
    },
    'solver': {
        'solver_type': 'linear', # linear solver with picard iteration
        'use_predictor_corrector': True, # for linear solver
        'n_corrector_steps': 10, # for linear solver
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
#        'log_iterations': False,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
# fmt: on


class IterHybridAgent(BaseAgent):  # noqa: D101
    def __init__(self, action_space):  # noqa: D107
        super().__init__(action_space=action_space)
        self.time = 0

    def act(self, observation) -> dict:  # noqa: D102
        action = {
            "Ip": [3e6],
            "NBI": [nbi_powers[0], nbi_cd[0], r_nbi, w_nbi],
            "ECRH": [eccd_power[0], 0.35, 0.05],
        }

        if self.time == 98:
            action["ECRH"][0] = eccd_power[99]
            action["NBI"][0] = nbi_powers[1]
            action["NBI"][1] = nbi_cd[1]

        if self.time >= 99:
            action["ECRH"][0] = eccd_power[100]
            action["NBI"][0] = nbi_powers[2]
            action["NBI"][1] = nbi_cd[2]

        if self.time < 99:
            action["Ip"][0] = 3e6 + (self.time + 1) * (12.5e6 - 3e6) / 100
        else:
            action["Ip"][0] = 12.5e6

        self.time += 1

        return action


class IterHybridEnv(BaseEnv):  # noqa: D101
    def __init__(self, render_mode, fig = None, store_state_history = False):  # noqa: D107
        super().__init__(
            render_mode=render_mode,
            log_level="debug",
            fig=fig,
            store_state_history=store_state_history,
        )

    @property
    def _define_actions(self):  # noqa: D102
        actions = [ah.IpAction(), ah.NbiAction(), ah.EcrhAction()]

        return actions

    @property
    def _define_observation(self):  # noqa: D102
        return oh.AllObservation()

    @property
    def _define_torax_config(self):  # noqa: D102
        return {
            "config": CONFIG,
            "discretization": "fixed",
            "ratio_a_sim": 1,
        }

    def _define_reward(self, state, next_state, action):  # noqa: D102
        return 0.0


if __name__ == "__main__":
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    fig_plot = None
    env = IterHybridEnv(render_mode=None, store_state_history=True)
    agent = IterHybridAgent(env.action_space)

    observation, _ = env.reset()
    terminated = False

    i = 0
    while not terminated:
        action = agent.act(observation)
        observation, _, terminated, _, _ = env.step(action)
        i += 1
        if i % 10 == 0:
            env.render()

    env.save_file("tmp/outputs_iter_torax.nc")
    # env.save_gif(filename="tmp/iter_hybrid.gif", interval=200, frame_step=7)
    viz.save_gif_from_nc(
        "tmp/outputs_iter_torax.nc",
        fig_properties=fig_plot,
        filename="tmp/output_torax.gif",
        interval=200,
        frame_step=5,
    )
