import numpy as np

from .. import rewards as reward
from ..action_handler import EcrhAction, IpAction, NbiAction
from ..observation_handler import AllObservation
from .base_env import BaseEnv

"""Config for ITER hybrid scenario based parameters with nonlinear solver.

ITER hybrid scenario based (roughly) on van Mulders Nucl. Fusion 2021.
With Newton-Raphson solver and adaptive timestep (backtracking)
"""

_NBI_W_TO_MA = 1 / 16e6  # rough estimate of NBI heating power to current drive
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
    "plasma_composition": {
        "main_ion": {"D": 0.5, "T": 0.5},  # (bundled isotope average)
        "impurity": {"Ne": 1 - W_to_Ne_ratio, "W": W_to_Ne_ratio},
        "Z_eff": {0.0: {0.0: 2.0, 1.0: 2.0}},  # sets impurity densities
    },
    "profile_conditions": {
        "Ip": {0: 3e6, 100: 12.5e6},  # total plasma current in MA
        "T_i": {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_i initial condition
        "T_i_right_bc": 0.2,  # T_i boundary condition
        "T_e": {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_e initial condition
        "T_e_right_bc": 0.2,  # T_e boundary condition
        "n_e_right_bc_is_fGW": True,
        "n_e_right_bc": {0: 0.35, 100: 0.35},  # n_e boundary condition
        # set initial condition density according to Greenwald fraction.
        "nbar": 0.85,  # line average density for initial condition
        "n_e": {0: {0.0: 1.3, 1.0: 1.0}},  # Initial electron density profile
        "normalize_n_e_to_nbar": True,  # normalize initial n_e to nbar
        "n_e_nbar_is_fGW": True,  # nbar is in units for greenwald fraction
        "initial_psi_from_j": True,  # initial psi from current formula
        "initial_j_is_total_current": True,  # only ohmic current on init
        "current_profile_nu": 2,  # exponent in initial current formula
    },
    "numerics": {
        "t_final": 150,  # length of simulation time in seconds
        "fixed_dt": 1,  # fixed timestep
        "evolve_ion_heat": True,  # solve ion heat equation
        "evolve_electron_heat": True,  # solve electron heat equation
        "evolve_current": True,  # solve current equation
        "evolve_density": True,  # solve density equation
    },
    "geometry": {
        "geometry_type": "chease",
        "geometry_file": "ITER_hybrid_citrin_equil_cheasedata.mat2cols",
        "Ip_from_parameters": True,
        "R_major": 6.2,  # major radius (R) in meters
        "a_minor": 2.0,  # minor radius (a) in meters
        "B_0": 5.3,  # Toroidal magnetic field on axis [T]
    },
    "sources": {
        # Current sources (for psi equation)
        "ecrh": {  # ECRH/ECCD (with Lin-Liu)
            "gaussian_width": 0.05,
            "gaussian_location": 0.35,
            "P_total": eccd_power,
        },
        "generic_heat": {  # Proxy for NBI heat source
            "gaussian_location": r_nbi,  # Gaussian location in normalized coordinates
            "gaussian_width": w_nbi,  # Gaussian width in normalized coordinates
            "P_total": (nbi_times, nbi_powers),  # Total heating power
            # electron heating fraction r
            "electron_heat_fraction": el_heat_fraction,
        },
        "generic_current": {  # Proxy for NBI current source
            "use_absolute_current": True,  # I_generic is total external current
            "gaussian_width": w_nbi,
            "gaussian_location": r_nbi,
            "I_generic": (nbi_times, nbi_cd),
        },
        "fusion": {},  # fusion power
        "ei_exchange": {},  # equipartition
        "ohmic": {},  # ohmic power
        "cyclotron_radiation": {},  # cyclotron radiation
        "impurity_radiation": {  # impurity radiation + bremsstrahlung
            "model_name": "mavrin_fit",
            "radiation_multiplier": 0.0,
        },
    },
    "neoclassical": {
        "bootstrap_current": {
            "bootstrap_multiplier": 1.0,
        },
    },
    "pedestal": {
        "model_name": "set_T_ped_n_ped",
        # use internal boundary condition model (for H-mode and L-mode)
        "set_pedestal": True,
        "T_i_ped": {0: 0.5, 100: 0.5, 105: 3.0},
        "T_e_ped": {0: 0.5, 100: 0.5, 105: 3.0},
        "n_e_ped_is_fGW": True,
        "n_e_ped": 0.85,  # pedestal top n_e in units of fGW
        "rho_norm_ped_top": 0.95,  # set ped top location in normalized radius
    },
    "transport": {
        "model_name": "qlknn",  # Using QLKNN_7_11 default
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        "apply_inner_patch": True,
        "D_e_inner": 0.15,
        "V_e_inner": 0.0,
        "chi_i_inner": 0.3,
        "chi_e_inner": 0.3,
        "rho_inner": 0.1,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        "apply_outer_patch": True,
        "D_e_outer": 0.1,
        "V_e_outer": 0.0,
        "chi_i_outer": 2.0,
        "chi_e_outer": 2.0,
        "rho_outer": 0.95,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        "chi_min": 0.05,  # minimum chi
        "chi_max": 100,  # maximum chi (can be helpful for stability)
        "D_e_min": 0.05,  # minimum electron diffusivity
        "D_e_max": 50,  # maximum electron diffusivity
        "V_e_min": -10,  # minimum electron convection
        "V_e_max": 10,  # minimum electron convection
        "smoothing_width": 0.1,
        "DV_effective": True,
        "include_ITG": True,  # to toggle ITG modes on or off
        "include_TEM": True,  # to toggle TEM modes on or off
        "include_ETG": True,  # to toggle ETG modes on or off
        "avoid_big_negative_s": False,
    },
    "solver": {
        "solver_type": "linear",  # linear solver with picard iteration
        "use_predictor_corrector": True,  # for linear solver
        "n_corrector_steps": 10,  # for linear solver
        "chi_pereverzev": 30,
        "D_pereverzev": 15,
        "use_pereverzev": True,
        #        'log_iterations': False,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}
# fmt: on


class IterHybridEnv(BaseEnv):
    """ITER hybrid scenario plasma control environment.

    This environment implements a plasma control task based on the ITER hybrid scenario,
    roughly following van Mulders Nucl. Fusion 2021. It provides control over plasma
    current (Ip), neutral beam injection (NBI), and electron cyclotron resonance heating
    (ECRH) with a comprehensive observation space including all available plasma parameters.

    The environment uses fixed time discretization with a configurable ratio between
    action timesteps and simulation timesteps. By default, it operates in debug logging
    mode for detailed monitoring of the simulation progress.

    Example:
        >>> # Basic usage with default parameters
        >>> env = IterHybridEnv()
        >>>
        >>> # Custom configuration
        >>> env = IterHybridEnv(
        ...     render_mode="human",
        ...     log_level="info",
        ...     store_history=True,
        ...     plot_config=plot_config
        ... )

    Note:
        All initialization parameters are passed to the base class `BaseEnv`.
        Refer to ``BaseEnv.__init__()`` documentation for detailed parameter descriptions.
    """

    def __init__(self, render_mode=None, **kwargs):
        """Initialize the ITER hybrid scenario environment.

        Args:
            render_mode (str, optional): Rendering mode for visualization.
                See `BaseEnv` documentation for details. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to ``BaseEnv.__init__()``.
                Common options include ``log_level``, ``log_file``, ``plot_config``, and ``store_history``.
                Refer to ``BaseEnv.__init__()`` documentation for complete parameter list.

        Note:
            This environment sets ``log_level`` to ``"debug"`` by default, which can be
            overridden by explicitly passing ``log_level`` in kwargs.
        """
        # Set environment-specific defaults
        kwargs.setdefault("log_level", "warning")
        kwargs.setdefault("plot_config", "default")

        super().__init__(render_mode=render_mode, **kwargs)

    def _define_action_space(self):  # noqa: D102
        actions = [
            IpAction(
                max=[15e6],  # 15 MA max plasma current
                ramp_rate=[0.2e6],
            ),  # 0.2 MA/s ramp rate limit
            NbiAction(
                max=[33e6, 1.0, 1.0],  # 33 MW max NBI power
            ),
            EcrhAction(
                max=[20e6, 1.0, 1.0],  # 20 MW max ECRH power
            ),
        ]

        return actions

    def _define_observation_space(self):  # noqa: D102
        return AllObservation(custom_bounds_file="gymtorax/envs/iter_hybrid.json")

    def _get_torax_config(self):  # noqa: D102
        return {
            "config": CONFIG,
            "discretization": "fixed",
            "ratio_a_sim": 1,
        }

    def _compute_reward(self, state, next_state, action):
        r"""Compute the reward. The higher the reward, the more performance and stability of the plasma.

        The reward is a weighted sum of several factors:
        - Fusion gain fusion_gain: we want to maximize it.
        - Beta_N: we want to be as close as possible to the Troyon limit, but not exceed it too much.
        - H-mode confinement quality factor h98: great if > 1
        - q_min: we want to avoid it to be below 1.
        - q_edge: we want to avoid it to be below 3.
        - Magnetic shear at rational surfaces: we want to avoid low shear at rational surfaces.

        Args:
            state (dict[str, Any]): state at time :math:`t`
            next_state (dict[str, Any]): state at time :math:`t+1`
            action (numpy.ndarray): applied action at time :math:`t`
            gamma (float): discounted factor (:math:`0 < \gamma \le 1`)
            n (int): number of steps since the beginning of the episode

        Returns:
            float: reward associated to the transition (state, action, next_state)
        """
        weight_list = [1, 1, 1, 1]

        def _is_H_mode():  # noqa: N802
            if (
                next_state["profiles"]["T_e"][0] > 10
                and next_state["profiles"]["T_i"][0] > 10
            ):
                return True
            else:
                return False

        def _r_fusion_gain():
            fusion_gain = (
                reward.get_fusion_gain(next_state) / 10
            )  # Normalize with ITER target
            if _is_H_mode():
                return fusion_gain
            else:
                return 0

        def _r_h98():
            h98 = reward.get_h98(next_state)
            if _is_H_mode():
                if h98 <= 1:
                    return h98
                else:
                    return 1
            else:
                return 0

        def _r_q_min():
            q_min = reward.get_q_min(next_state)
            if q_min <= 1:
                return q_min
            elif q_min > 1:
                return 1

        def _r_q_95():
            q_95 = reward.get_q95(next_state)
            if q_95 / 3 <= 1:
                return q_95 / 3
            else:
                return 1

        # Calculate individual reward components
        r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
        r_h98 = weight_list[1] * _r_h98() / 50
        r_q_min = weight_list[2] * _r_q_min() / 150
        r_q_95 = weight_list[3] * _r_q_95() / 150

        total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_95

        # Store reward breakdown for logging (attach to environment if it has reward_breakdown attribute)
        if hasattr(self, "reward_breakdown"):
            if not hasattr(self, "_reward_components"):
                self._reward_components = {
                    "fusion_gain": [],
                    "h98": [],
                    "q_min": [],
                    "q_edge": [],
                }

            self._reward_components["fusion_gain"].append(r_fusion_gain)
            self._reward_components["h98"].append(r_h98)
            self._reward_components["q_min"].append(r_q_min)
            self._reward_components["q_edge"].append(r_q_95)

        return total_reward
