import numpy as np


def get_fusion_gain(state: dict) -> float:
    """Get the fusion gain Q from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The fusion gain Q.
    """
    return state["scalars"]["Q_fusion"][0]


def get_beta_N(state: dict) -> float:  # noqa: N802
    """Get the normalized beta_N from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The normalized beta_N.
    """
    return state["scalars"]["beta_N"][0]


def get_tau_E(state: dict) -> float:  # noqa: N802
    """Get the energy confinement time tau_E from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The energy confinement time tau_E.
    """
    return state["scalars"]["tau_E"][0]


def get_h98(state: dict) -> float:  # noqa: N802
    """Get the H-mode confinement quality factor from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The H98 factor.
    """
    return state["scalars"]["H98"][0]


def get_q_profile(state: dict) -> np.ndarray:
    """Get the safety factor profile q from the state dictionary.

    Args:
        state (dict): The state dictionary containing profile values.

    Returns:
        np.ndarray: The safety factor profile q.
    """
    return state["profiles"]["q"]


def get_q_min(state: dict) -> float:
    """Get the minimum safety factor q_min from the state dictionary.

    Args:
        state (dict): The state dictionary containing scalar values.

    Returns:
        float: The minimum safety factor q_min.
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
    """Get the magnetic shear profile s from the state dictionary.

    Args:
        state (dict): The state dictionary containing profile values.

    Returns:
        np.ndarray: The magnetic shear profile s.
    """
    return state["profiles"]["magnetic_shear"]
