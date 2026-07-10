"""Scenario regression tests: gymtorax rollout vs native TORAX ground truth.

This module runs a full physics scenario through the gymtorax wrapper and
compares its per-timestep output against a reference produced by running the
*same* TORAX configuration through native TORAX. It is the only test that
validates the *correctness of the physics results* produced by the wrapper:
if gymtorax's action injection and state extraction faithfully reproduce the
prescribed scenario, the profiles and scalars must match native TORAX to
float precision.

The reference is a git-tracked NetCDF file per scenario, stored in
``tests/scenario_references/``. It is (re)generated with::

    pytest --generate-references tests/test_scenarios.py

which runs *only* ``test_generate_references``. The comparison tests are opt-in
(they run a full, slow simulation) and are the only ones that run under::

    pytest --test-scenarios tests/test_scenarios.py

Both flags are defined in ``tests/conftest.py``, which also handles selecting the
right tests for each flag.

Tolerance rationale
-------------------
The only *expected* difference between gymtorax and native TORAX is that
gymtorax casts its observations to ``float32`` while TORAX computes in
``float64``. A single float32 round-trip has a relative error of at most
half a ULP (``2**-24``). We therefore use ``rtol = float32 machine epsilon``
(``2**-23`` ~ 1.19e-7), which is one ULP and leaves a half-ULP safety
margin, with ``atol = 0`` for a pure relative check.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torax
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output

from gymtorax import IterHybridAgent, IterHybridEnv
from gymtorax.envs import iter_hybrid_env

# Directory holding the git-tracked reference files.
REF_DIR = Path(__file__).parent / "scenario_references"

# Pure relative tolerance at the float32 precision floor (see module docstring).
RTOL = float(np.finfo(np.float32).eps)  # 2**-23 ~ 1.19e-7
ATOL = 0.0

# Scalars whose initial (t=0) value is a known, accepted difference and is
# excluded from the comparison. ``v_loop_lcfs`` (= dpsi_lcfs/dt) is undefined at
# the initial state for the Ip boundary-condition case: TORAX initializes it to
# 0 and then, only when assembling the *full* output, back-fills
# ``v_loop_lcfs[0] = v_loop_lcfs[1]`` because the series is underconstrained
# (see torax output.py / initialization.py). gymtorax builds each state's output
# one step at a time, so it never sees index [1] at reset and legitimately
# observes the raw 0.0.
KNOWN_INITIAL_STATE_DIFFS = frozenset({"v_loop_lcfs"})


@dataclass(frozen=True)
class ScenarioSpec:
    """Definition of a comparable scenario.

    Attributes:
        config: TORAX configuration dict, used to build the native reference.
        env_factory: Builds the gymtorax environment for the rollout.
        agent_factory: Builds the scripted agent, given the env action space.
    """

    config: dict[str, Any]
    env_factory: Callable[[], Any]
    agent_factory: Callable[[Any], Any]


# Registry of scenarios. Add a scenario by adding one entry here.
SCENARIOS: dict[str, ScenarioSpec] = {
    "iter_hybrid": ScenarioSpec(
        config=iter_hybrid_env.CONFIG,
        env_factory=lambda: IterHybridEnv(log_level="warning"),
        agent_factory=lambda action_space: IterHybridAgent(action_space),
    ),
}


@dataclass
class Rollout:
    """Per-timestep gymtorax output collected during a scenario rollout."""

    times: np.ndarray
    profiles: dict[str, np.ndarray]  # var -> [T, n]
    scalars: dict[str, np.ndarray]  # var -> [T]


# =============================================================================
# Reference generation and loading
# =============================================================================


def _reference_path(name: str) -> Path:
    return REF_DIR / f"{name}.nc"


def _generate_reference(name: str, spec: ScenarioSpec) -> None:
    """Run the scenario through native TORAX and store the output as NetCDF."""
    torax_config = torax.ToraxConfig.from_dict(spec.config)
    output_xr, _ = run_simulation.run_simulation(torax_config, progress_bar=False)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    output_xr.to_netcdf(_reference_path(name), engine="h5netcdf")


def _load_reference(name: str):
    """Load the native-TORAX reference DataTree for a scenario."""
    path = _reference_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing reference {path}. Generate it with "
            f"`pytest --generate-references tests/test_scenarios.py`."
        )
    return output.load_state_file(str(path))


# =============================================================================
# gymtorax rollout
# =============================================================================


def _rollout_gymtorax(spec: ScenarioSpec) -> Rollout:
    """Roll out the scenario through gymtorax and collect per-step observations."""
    env = spec.env_factory()
    try:
        agent = spec.agent_factory(env.action_space)
        observation, _ = env.reset(seed=0)

        times: list[float] = [env.current_time]
        profiles: dict[str, list[np.ndarray]] = defaultdict(list)
        scalars: dict[str, list[np.ndarray]] = defaultdict(list)

        def _record(obs: dict[str, dict[str, np.ndarray]]) -> None:
            for var, value in obs["profiles"].items():
                profiles[var].append(np.asarray(value))
            for var, value in obs["scalars"].items():
                scalars[var].append(np.asarray(value))

        _record(observation)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            _record(observation)
            times.append(env.current_time)
    finally:
        env.close()

    return Rollout(
        times=np.asarray(times),
        profiles={var: np.stack(vals) for var, vals in profiles.items()},
        # scalars are stored as length-1 arrays per step; squeeze to [T].
        scalars={
            var: np.stack(vals).reshape(len(vals)) for var, vals in scalars.items()
        },
    )


# =============================================================================
# Reference generation test
# =============================================================================
#
# Collection is split by flag in conftest.py: ``--generate-references`` runs
# ONLY ``test_generate_references`` (the comparison tests are deselected), while
# ``--test-scenarios`` runs ONLY the comparison tests (this one is deselected).


@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_generate_references(name):
    """Regenerate the native-TORAX reference for a scenario.

    Review the resulting git diff, commit it, then verify with
    ``pytest --test-scenarios``.
    """
    _generate_reference(name, SCENARIOS[name])
    assert _reference_path(name).exists()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session", params=list(SCENARIOS), ids=list(SCENARIOS))
def scenario(request):
    """Provide (name, gymtorax rollout, native-TORAX reference) for a scenario.

    Rolled out once per scenario and shared across the category subtests.
    """
    name = request.param
    spec = SCENARIOS[name]
    reference = _load_reference(name)
    rollout = _rollout_gymtorax(spec)
    return name, rollout, reference


# =============================================================================
# Comparison helpers
# =============================================================================


def _ref_profiles(reference):
    return reference.children[output.PROFILES].dataset


def _ref_scalars(reference):
    return reference.children[output.SCALARS].dataset


def _ref_variable(ref_ds, var: str) -> np.ndarray | None:
    """Resolve an observed variable name to its reference array.

    gymtorax's observation handler splits TORAX variables that carry a leading
    species dimension (e.g. ``main_ion_fractions`` with ``main_ion=['D','T']``)
    into per-species, time-leading variables named ``{var}_{symbol}`` (e.g.
    ``main_ion_fractions_D``). The native reference stores them unsplit, so
    such names are resolved by slicing the species coordinate and moving time
    to the leading axis. Returns ``None`` if the name cannot be resolved.
    """
    if var in ref_ds:
        return ref_ds[var].to_numpy()
    for name, da in ref_ds.data_vars.items():
        if da.dims and da.dims[0] != "time" and var.startswith(f"{name}_"):
            label = var[len(name) + 1 :]
            species_dim = da.dims[0]
            labels = [str(v) for v in da[species_dim].values]
            if label in labels:
                return (
                    da.isel({species_dim: labels.index(label)})
                    .transpose("time", ...)
                    .to_numpy()
                )
    return None


def _align_ref(actual: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align a reference array to the gymtorax time series.

    Some reference "scalars" are time-independent constants (e.g. the grid
    spacing ``drho_norm``), stored 0-dimensional by native TORAX. gymtorax
    instead observes them once per step, i.e. as a constant ``[T]`` series.
    Broadcast such constants so they compare step-by-step.
    """
    ref = np.asarray(ref)
    if ref.ndim == 0:
        return np.broadcast_to(ref, actual.shape)
    return ref


def _compare_timeseries(
    name: str,
    actual: np.ndarray,
    ref: np.ndarray,
    times: np.ndarray,
    start: int = 0,
) -> None:
    """Compare an ``[T, ...]`` array timestep-by-timestep against a reference.

    Reports the first mismatching step and the final (steady-state) step, plus
    a per-step relative-error-norm trace, mirroring the TORAX sim test.

    Args:
        start: First timestep to compare, used to skip known initial-state
            differences (see ``KNOWN_INITIAL_STATE_DIFFS``).
    """
    ref = _align_ref(actual, ref)
    n_steps = min(actual.shape[0], ref.shape[0])
    err_norms = []
    first_mismatch: str | None = None

    for step in range(start, n_steps):
        a = np.atleast_1d(actual[step]).astype(np.float64).ravel()
        r = np.atleast_1d(ref[step]).astype(np.float64).ravel()
        denom = np.where(r == 0.0, np.finfo(np.float64).eps, r)
        err_norms.append(float(np.sqrt(np.square((a - r) / denom).sum())))

        if first_mismatch is None and not np.allclose(a, r, rtol=RTOL, atol=ATOL):
            lines = [
                f"'{name}': mismatch at step {step} (t = {times[step]})",
                "Pos\tActual\tExpected\tRelErr\tMatch",
            ]
            for i in range(a.shape[0]):
                match = np.allclose(a[i], r[i], rtol=RTOL, atol=ATOL)
                rele = np.abs((a[i] - r[i]) / denom[i])
                lines.append(f"{i}\t{a[i]:.9g}\t{r[i]:.9g}\t{rele:.3e}\t{match}")
            first_mismatch = "\n".join(lines)

    if first_mismatch is not None:
        trace = "\n".join(f"\t{i}\t{e:.6e}" for i, e in enumerate(err_norms))
        raise AssertionError(
            f"{first_mismatch}\n\n"
            f"Final step {n_steps - 1} rel-error norm: {err_norms[-1]:.6e}\n"
            f"Rel-error norm over time:\n\tStep\tNorm\n{trace}"
        )


# =============================================================================
# Category subtests
# =============================================================================


def test_scenario_structure(scenario):
    """1) Shape/length: step count and per-variable array shapes match TORAX."""
    name, rollout, reference = scenario
    ref_profiles = _ref_profiles(reference)
    ref_scalars = _ref_scalars(reference)
    ref_time = ref_profiles[output.TIME].to_numpy()

    # Number of timesteps (including the initial state at t=0).
    assert rollout.times.shape[0] == ref_time.shape[0], (
        f"'{name}': gymtorax produced {rollout.times.shape[0]} states, "
        f"TORAX reference has {ref_time.shape[0]}."
    )

    # Every observed variable must exist in the reference with matching shape.
    for var, arr in rollout.profiles.items():
        ref_arr = _ref_variable(ref_profiles, var)
        assert ref_arr is not None, f"'{name}': profile '{var}' absent from reference."
        assert arr.shape == ref_arr.shape, (
            f"'{name}': profile '{var}' shape {arr.shape} != reference {ref_arr.shape}."
        )

    for var, arr in rollout.scalars.items():
        ref_arr = _ref_variable(ref_scalars, var)
        assert ref_arr is not None, f"'{name}': scalar '{var}' absent from reference."
        assert arr.shape == (rollout.times.shape[0],), (
            f"'{name}': scalar '{var}' has shape {arr.shape}, expected one value "
            f"per state ({rollout.times.shape[0]},)."
        )
        # Time-independent constants (e.g. grid spacing) are stored 0-d by
        # TORAX but observed once per step by gymtorax; only time-series scalars
        # must match the reference length.
        if ref_arr.ndim > 0:
            assert arr.shape[0] == ref_arr.shape[0], (
                f"'{name}': scalar '{var}' length {arr.shape[0]} != reference "
                f"{ref_arr.shape[0]}."
            )


def test_scenario_metadata(scenario):
    """2) Metadata: time axis and variable coverage agree with TORAX."""
    name, rollout, reference = scenario
    ref_profiles = _ref_profiles(reference)
    ref_time = ref_profiles[output.TIME].to_numpy()

    # The action-controlled variables are excluded from the observation, so
    # every observed variable must resolve to a reference variable (directly
    # or as a species-split slice; see `_ref_variable`).
    unresolved_profiles = [
        var for var in rollout.profiles if _ref_variable(ref_profiles, var) is None
    ]
    assert not unresolved_profiles, (
        f"'{name}': observed profiles absent from reference: {unresolved_profiles}."
    )
    ref_scalars = _ref_scalars(reference)
    unresolved_scalars = [
        var for var in rollout.scalars if _ref_variable(ref_scalars, var) is None
    ]
    assert not unresolved_scalars, (
        f"'{name}': observed scalars absent from reference: {unresolved_scalars}."
    )

    # The simulation time axis must line up step-for-step.
    np.testing.assert_allclose(
        rollout.times,
        ref_time,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"'{name}': simulation time axis differs from reference.",
    )


def test_scenario_scalars(scenario):
    """3) Scalar values: every scalar observation matches TORAX per timestep."""
    name, rollout, reference = scenario
    ref_scalars = _ref_scalars(reference)
    ref_time = _ref_profiles(reference)[output.TIME].to_numpy()

    assert rollout.scalars, f"'{name}': rollout produced no scalar observations."
    for var, arr in rollout.scalars.items():
        start = 1 if var in KNOWN_INITIAL_STATE_DIFFS else 0
        ref_arr = _ref_variable(ref_scalars, var)
        assert ref_arr is not None, f"'{name}': scalar '{var}' absent from reference."
        _compare_timeseries(
            f"{name}:scalars:{var}",
            arr,
            ref_arr,
            ref_time,
            start=start,
        )


def test_scenario_profiles(scenario):
    """4) Profile arrays: every profile observation matches TORAX per timestep."""
    name, rollout, reference = scenario
    ref_profiles = _ref_profiles(reference)
    ref_time = ref_profiles[output.TIME].to_numpy()

    assert rollout.profiles, f"'{name}': rollout produced no profile observations."
    for var, arr in rollout.profiles.items():
        ref_arr = _ref_variable(ref_profiles, var)
        assert ref_arr is not None, f"'{name}': profile '{var}' absent from reference."
        _compare_timeseries(f"{name}:profiles:{var}", arr, ref_arr, ref_time)
