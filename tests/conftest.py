import gymnasium as gym
import pytest

ALL_ENV_IDS = ["gymtorax/IterHybrid-v0", "gymtorax/Test-v0"]


@pytest.fixture
def make_env():
    envs = []

    def _make(env_id):
        env = gym.make(env_id)
        envs.append(env)
        return env

    yield _make

    for env in envs:
        env.close()


def pytest_addoption(parser):
    parser.addoption(
        "--docs", action="store_true", default=False, help="Run documentation tests"
    )
    parser.addoption(
        "--test-scenarios",
        action="store_true",
        default=False,
        help="Run the slow scenario regression tests (gymtorax vs native TORAX)",
    )
    parser.addoption(
        "--generate-references",
        action="store_true",
        default=False,
        help=(
            "Dedicated mode: run ONLY the scenario reference-generation test "
            "(regenerate the stored native-TORAX references) and nothing else."
        ),
    )


def _is_scenario_generator(item) -> bool:
    return (
        item.fspath.basename == "test_scenarios.py"
        and "test_generate_references" in item.nodeid
    )


def _is_scenario_comparison(item) -> bool:
    return (
        item.fspath.basename == "test_scenarios.py"
        and "test_generate_references" not in item.nodeid
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--docs"):
        skip = pytest.mark.skip(reason="pass --docs to run documentation tests")
        for item in items:
            if item.fspath.basename == "test_docs.py":
                item.add_marker(skip)

    # --generate-references is a dedicated mode: run ONLY the reference-generation
    # test(s), deselecting the entire rest of the suite (incl. the slow env
    # checkers), regardless of the paths pytest was pointed at.
    if config.getoption("--generate-references"):
        keep = [item for item in items if _is_scenario_generator(item)]
        drop = [item for item in items if not _is_scenario_generator(item)]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
        return

    # Not generating. The generation test never runs here; scenario comparison
    # tests are opt-in via --test-scenarios and otherwise skipped.
    run_scenarios = config.getoption("--test-scenarios")
    skip = pytest.mark.skip(
        reason="pass --test-scenarios (or --generate-references) to run scenario tests"
    )
    keep, drop = [], []
    for item in items:
        if _is_scenario_generator(item):
            drop.append(item)  # generation only runs under --generate-references
            continue
        if _is_scenario_comparison(item) and not run_scenarios:
            item.add_marker(skip)
        keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
