"""TORAX reward module.

This module provides functions to extract specific metrics from the state
dictionary returned by the TORAX simulator. These metrics can be used to
construct reward functions for reinforcement learning environments focused
on tokamak control and optimization. Other reward functions can be created.
"""

import numpy as np


def get_fusion_gain(state: dict) -> float:
    """Get the fusion gain :math:`Q` from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The fusion gain :math:`Q`.
    """
    return state["scalars"]["Q_fusion"][0]


def get_beta_N(state: dict) -> float:  # noqa: N802
    r"""Get the normalized :math:`\beta_N` from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The normalized :math:`\beta_N`.
    """
    return state["scalars"]["beta_N"][0]


def get_tau_E(state: dict) -> float:  # noqa: N802
    r"""Get the energy confinement time :math:`\tau_E` from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The energy confinement time :math:`\tau_E`.
    """
    return state["scalars"]["tau_E"][0]


def get_h98(state: dict) -> float:  # noqa: N802
    """Get the H-mode confinement quality factor from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The :math:`H98` factor.
    """
    return state["scalars"]["H98"][0]


def get_q_profile(state: dict) -> np.ndarray:
    """Get the safety factor profile :math:`q` from the state dictionary.

    Args:
        state (dict): The state dictionary containing profile values.

    Returns:
        numpy.ndarray: The safety factor profile :math:`q`.
    """
    return state["profiles"]["q"]


def get_q_min(state: dict) -> float:
    """Get the minimum safety factor :math:`q_{min}` from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The minimum safety factor :math:`q_{min}`.
    """
    return state["scalars"]["q_min"][0]


def get_q95(state: dict) -> float:
    """Get safety factor at 95% of the normalized poloidal flux coordinate.

    Args:
        state (dict): The state dictionary containing profile values.

    Returns:
        float: The safety factor at 95% of the normalized poloidal flux coordinate.
    """
    return state["scalars"]["q95"][0]


def get_s_profile(state: dict) -> np.ndarray:
    """Get the magnetic shear profile :math:`s` from the state dictionary.

    Args:
        state (dict): The state dictionary containing profile values.

    Returns:
        numpy.ndarray: The magnetic shear profile :math:`s`.
    """
    return state["profiles"]["magnetic_shear"]
