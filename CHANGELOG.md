# Changelog

All notable changes to GymTORAX are documented in this file.

## [1.1.0] - 2026-07-11

This release upgrades GymTORAX from TORAX 1.0 to TORAX 1.4, rebuilds the simulation core on the new TORAX orchestration API, and adds three new actions, restart-from-file support, and a physics regression test suite. Several aspects of the wrapper internal logic are also improved.

### Added

- **`GenericHeatAction`** - control of the TORAX `generic_heat` source (total power, deposition location, deposition width) without driving current, for configs where `generic_current` is prescribed (e.g. fraction-of-Ip mode).
- **`GasPuffAction`** - gas puff fueling control (total particle rate, edge decay length).
- **`PelletAction`** - pellet fueling control (particle rate, deposition location, deposition width). TORAX models pellets as a continuous time-averaged source, so the action sets an average fueling rate rather than discrete pellet events.
- **Restart from a previous TORAX run** - episodes can start from a saved simulation via `restart.do_restart` in the environment config. The config is validated against the restart file at load time (initial action values must match the file, and the psi initialization mode must let TORAX restore psi from the file).
- **Stricter config validation at load time** - action-controlled config keys must hold a single initial value, and `NbiAction` combined with `generic_current` requires `use_absolute_current: True`.
- **More complete tests** - `pytest --test-scenarios` replays a full ITER hybrid episode through the Gymnasium API and checks it against a native TORAX reference at single-float precision; `pytest --generate-references` regenerates the references after a TORAX upgrade. Also added tests for all the other added features

### Changed

- **TORAX 1.4 support** - the `torax` dependency moves from `1.0.*` to `1.4.*`. Python 3.11+ is now required (TORAX 1.4 requirement).
- **Action semantics: ramp-to-setpoint** - action values now ramp linearly from the current value to the setpoint over each action window, instead of being applied as a constant hold.
- **Observations** - output variables with a species dimension (new in TORAX 1.4) are split into per-species entries named `<variable>_<species>`.

### Performance

- Simulation output is converted to observation data once per step instead of twice, reducing gymtorax overhead.
- Any optimization brought by the latest version of TORAX.

### Fixed

- The reward function now receives the previous state as the documented dictionary format instead of a raw xarray `DataTree`.
- The Ip consistency check in `ConfigLoader` now looks up scalar action variables correctly.

## [1.0.0] - 2025-10-09

Initial public release: Gymnasium environments on top of TORAX 1.0 with the
ITER hybrid ramp-up scenario (`gymtorax/IterHybrid-v0`), configurable action
and observation handlers, rendering, and baseline agents.

[1.1.0]: https://github.com/antoine-mouchamps/gymtorax/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/antoine-mouchamps/gymtorax/releases/tag/v1.0.0
