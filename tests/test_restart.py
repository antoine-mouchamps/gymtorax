"""Restart feature tests: consistency guards and functional restart.

When an episode restarts from a previous TORAX run (``restart.do_restart``),
TORAX restores the plasma state from the file but evaluates all
action-controlled parameters from the configuration. The configuration must
therefore provide, for every such parameter, its value at the restart time
as recorded in the file; ``ConfigLoader._validate`` raises otherwise (see
``ActionHandler.validate_restart_scalars``).

These tests use the git-tracked scenario reference
``tests/scenario_references/iter_hybrid.nc`` as the restart source. Expected
values are always read from the file itself, so the tests survive reference
regeneration (``pytest --generate-references``).

Like the scenario comparison tests, this module only runs under
``pytest --test-scenarios`` (see ``conftest.py``). Most tests here are fast
consistency-guard checks; the functional test at the end restarts an actual
episode from the reference file and verifies the loaded plasma state, the
episode clock, and stepping.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gymtorax.action_handler import (
    ActionHandler,
    EcrhAction,
    IpAction,
    NbiAction,
)
from gymtorax.envs import iter_hybrid_env
from gymtorax.envs.iter_hybrid_env import IterHybridEnv
from gymtorax.torax_wrapper.config_loader import ConfigLoader

REF_FILE = Path(__file__).parent / "scenario_references" / "iter_hybrid.nc"
RESTART_TIME = 50.0

pytestmark = pytest.mark.skipif(
    not REF_FILE.exists(),
    reason=(
        "scenario reference file missing; generate it with "
        "`pytest --generate-references tests/test_scenarios.py`"
    ),
)


@pytest.fixture(scope="module")
def file_state():
    """Snapshot of the reference file at the restart time."""
    with open(REF_FILE, "rb") as f:
        tree = xr.open_datatree(f).compute()
    tree = tree.sel(time=RESTART_TIME, method="nearest")
    return {
        "scalars": tree.children["scalars"].dataset.squeeze(),
        "profiles": tree.children["profiles"].dataset.squeeze(),
    }


def _make_restart_config(
    file_state,
    ip=None,
    time=RESTART_TIME,
    do_restart=True,
    filename=str(REF_FILE),
    t_final=55.0,
):
    """IterHybrid config restarting from the reference file.

    Action-controlled parameters get single initial values taken from the file
    at the restart time (overridable for the mismatch tests).
    """
    scalars = file_state["scalars"]
    p_nbi = float(scalars["P_aux_generic_total"])

    config = copy.deepcopy(iter_hybrid_env.CONFIG)
    config["restart"] = {
        "filename": filename,
        "time": time,
        "do_restart": do_restart,
        "stitch": False,
    }
    config["numerics"]["t_final"] = t_final
    # The fresh-start config computes the initial psi from the current formula
    # (initial_psi_mode='j'); a restart must instead honor the psi restored
    # from the file, which TORAX does only in 'profile_conditions' mode.
    config["profile_conditions"]["initial_psi_mode"] = "profile_conditions"
    config["profile_conditions"]["Ip"] = (
        ip if ip is not None else float(scalars["Ip"])
    )
    config["sources"]["ecrh"]["P_total"] = float(scalars["P_ecrh_e"])
    config["sources"]["generic_heat"]["P_total"] = p_nbi
    config["sources"]["generic_current"]["I_generic"] = p_nbi / 16e6
    return config


def _make_handler():
    return ActionHandler([IpAction(), NbiAction(), EcrhAction()])


# =============================================================================
# Consistency check guards (fast, no simulation)
# =============================================================================


def test_restart_matching_config_accepted(file_state):
    # Initial values matching the file at the restart time construct cleanly.
    ConfigLoader(_make_restart_config(file_state), _make_handler())


def test_restart_mismatch_raises(file_state):
    # A wrong initial Ip must be rejected at construction.
    config = _make_restart_config(file_state, ip=3e6)
    with pytest.raises(ValueError, match="Restart consistency"):
        ConfigLoader(config, _make_handler())


def test_restart_check_skipped_when_do_restart_false(file_state):
    # Same wrong Ip, but do_restart=False: the check must not fire.
    config = _make_restart_config(file_state, ip=3e6, do_restart=False)
    ConfigLoader(config, _make_handler())


def test_restart_missing_file_raises(file_state):
    config = _make_restart_config(file_state, filename="/nonexistent/output.nc")
    with pytest.raises(ValueError, match="Could not load restart file"):
        ConfigLoader(config, _make_handler())


def test_restart_nearest_snapshot_used(file_state):
    # A restart time between snapshots compares against the nearest one,
    # mirroring TORAX's own `method="nearest"` selection.
    config = _make_restart_config(file_state, time=RESTART_TIME + 0.4)
    ConfigLoader(config, _make_handler())


def test_restart_rejects_initial_psi_mode_j(file_state):
    # TORAX only honors the psi restored from the file in
    # 'profile_conditions' mode; any other mode silently recomputes the
    # initial psi, so it must be rejected on restart.
    config = _make_restart_config(file_state)
    config["profile_conditions"]["initial_psi_mode"] = "j"
    with pytest.raises(ValueError, match="initial_psi_mode"):
        ConfigLoader(config, _make_handler())


# =============================================================================
# Functional restart (slow, runs a real simulation)
# =============================================================================


def test_restart_episode_runs_from_file(file_state):
    """A restarted episode starts from the file state and steps to t_final."""
    restart_config = _make_restart_config(file_state, t_final=55.0)
    scalars = file_state["scalars"]

    class RestartIterHybridEnv(IterHybridEnv):
        def _get_torax_config(self):  # noqa: D102
            return {
                "config": restart_config,
                "discretization": "fixed",
                "ratio_a_sim": 1,
            }

    env = RestartIterHybridEnv(log_level="warning")
    try:
        observation, _ = env.reset(seed=0)

        # The episode clock starts at the restart time.
        assert env.current_time == pytest.approx(RESTART_TIME)

        # The initial plasma state comes from the file, not from the config
        # initial conditions (which describe the t=0 plasma of the original
        # run and differ wildly from the evolved state at t=50).
        ref_profiles = file_state["profiles"]
        for var in ["T_e", "T_i", "n_e", "psi"]:
            np.testing.assert_allclose(
                np.asarray(observation["profiles"][var], dtype=np.float64),
                ref_profiles[var].to_numpy(),
                rtol=1e-4,
                err_msg=f"initial profile '{var}' does not match the restart file",
            )

        # Step to t_final with constant actions held at the restart values.
        action = {
            "Ip": np.array([float(scalars["Ip"])]),
            "NBI": np.array([float(scalars["P_aux_generic_total"]), 0.25, 0.25]),
            "ECRH": np.array([float(scalars["P_ecrh_e"]), 0.35, 0.05]),
        }
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(action)
            steps += 1
            assert steps <= 10, "episode did not terminate at t_final"

        assert terminated
        assert steps == 5  # 50 -> 55 s in 1 s action windows
        assert env.current_time == pytest.approx(55.0)
    finally:
        env.close()
