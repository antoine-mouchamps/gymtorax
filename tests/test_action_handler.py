import logging

import numpy as np
import pytest
from torax._src.core_profiles.profile_conditions import _MIN_IP_AMPS

from gymtorax.action_handler import (
    Action,
    ActionHandler,
    EcrhAction,
    GasPuffAction,
    IpAction,
    NbiAction,
    VloopAction,
)

# ------------------------
# Custom Actions for testing
# ------------------------


class CustomAction(Action):
    name = "CustomAction"
    dimension = 2
    default_min = [0.0, -1.0]
    default_max = [10.0, 1.0]
    default_ramp_rate = [None, None]
    config_mapping = {
        ("some_config", "param1"): (0, 1),
        ("some_config", "param2"): (1, 1),
    }
    state_var = {"scalars": ["param1", "param2"]}


class CustomAction1(Action):
    name = "CustomAction1"
    dimension = 1
    default_min = [0.0]
    default_max = [1.0]
    default_ramp_rate = [None]
    config_mapping = {("some_config", "param1"): (0, 1)}
    state_var = {"scalars": ["param1"]}


class CustomAction2(Action):
    name = "CustomAction2"
    dimension = 1
    default_min = [0.0]
    default_max = [2.0]
    default_ramp_rate = [None]
    config_mapping = {("some_config", "param2"): (0, 1)}
    state_var = {"scalars": ["param2"]}


# ------------------------
# Action class tests
# ------------------------


def test_action_init_defaults():
    # Test default initialization of CustomAction
    action = CustomAction()
    np.testing.assert_array_equal(action.min, [0.0, -1.0])
    np.testing.assert_array_equal(action.max, [10.0, 1.0])
    np.testing.assert_array_equal(action.values, [0.0, -1.0])


def test_action_init_custom_bounds():
    # Test CustomAction initialization with custom min/max bounds
    action = CustomAction(min=[1.0, 0.0], max=[5.0, 0.5])
    np.testing.assert_array_equal(action.min, [1.0, 0.0])
    np.testing.assert_array_equal(action.max, [5.0, 0.5])
    np.testing.assert_array_equal(action.values, [1.0, 0.0])


def test_action_invalid_dimension():
    # Test that Action with invalid dimension raises ValueError
    class BadAction(Action):
        name = "BadAction"
        dimension = 0
        default_min = []
        default_max = []
        config_mapping = {}

    with pytest.raises(ValueError):
        BadAction()


def test_action_invalid_min_length():
    # Test that Action with invalid default_min length raises ValueError
    class BadAction(Action):
        name = "BadAction"
        dimension = 2
        default_min = [0.0]
        default_max = [1.0, 2.0]
        config_mapping = {("a",): 0, ("b",): 1}

    with pytest.raises(ValueError):
        BadAction()


def test_action_invalid_max_length():
    # Test that Action with invalid default_max length raises ValueError
    class BadAction(Action):
        name = "BadAction"
        dimension = 2
        default_min = [0.0, 1.0]
        default_max = [1.0]
        config_mapping = {("a",): 0, ("b",): 1}

    with pytest.raises(ValueError):
        BadAction()


def test_action_set_values_valid():
    # Test setting valid values for CustomAction
    action = CustomAction()
    action._set_values([5.0, 0.5])
    np.testing.assert_array_equal(action.values, [5.0, 0.5])


def test_action_set_values_invalid_length():
    # Test that setting values with wrong length raises ValueError
    action = CustomAction()
    with pytest.raises(ValueError):
        action._set_values([1.0])


def test_action_repr():
    # Test string representation of CustomAction
    action = CustomAction()
    r = repr(action)
    assert "CustomAction" in r
    assert "values" in r
    assert "min" in r
    assert "max" in r


def test_action_get_mapping():
    # Test retrieval of config_mapping from CustomAction
    action = CustomAction()
    mapping = action.get_mapping()
    assert isinstance(mapping, dict)
    assert ("some_config", "param1") in mapping


def test_action_init_dict_and_provider_updates():
    # Test init_dict and get_provider_updates methods for CustomAction
    action = CustomAction()
    action._set_values([2.0, 0.5])
    config = {"some_config": {"param1": None, "param2": None}}
    action.init_dict(config)
    assert config["some_config"]["param1"][0][0] == 2.0
    assert config["some_config"]["param2"][0][0] == 0.5
    # Update at time=1.0: ramps from the previous value (at window start) to
    # the new value (at window end).
    action._set_values([3.0, 0.7])
    updates = action.get_provider_updates(time=1.0, delta_t_a=1.0)
    assert updates["some_config.param1"].time.tolist() == [1.0, 2.0]
    assert updates["some_config.param1"].value.tolist() == [2.0, 3.0]
    assert updates["some_config.param2"].time.tolist() == [1.0, 2.0]
    assert updates["some_config.param2"].value.tolist() == [0.5, 0.7]


def test_action_init_dict_keyerror():
    # Test that init_dict raises RuntimeError for missing config structure
    action = CustomAction()
    config = {}  # Missing structure
    with pytest.raises(RuntimeError):
        action.init_dict(config)


def test_action_init_dict_rejects_time_varying_dict():
    # A time-varying series (dict with several time nodes) for an
    # action-controlled key must be rejected: only the t=0 value is allowed.
    action = CustomAction1()
    config = {"some_config": {"param1": {0: 0.2, 100: 0.8}}}
    with pytest.raises(RuntimeError, match="time-varying"):
        action.init_dict(config)


def test_action_init_dict_rejects_time_varying_tuple():
    # Same rejection for the (times, values) tuple form.
    action = CustomAction1()
    config = {"some_config": {"param1": (np.array([0, 100]), np.array([0.2, 0.8]))}}
    with pytest.raises(RuntimeError, match="time-varying"):
        action.init_dict(config)


def test_action_init_dict_accepts_single_time_node():
    # A single value at t=0 is fine, in scalar, dict, or tuple form.
    for initial in [0.4, {0: 0.4}, (np.array([0]), np.array([0.4]))]:
        action = CustomAction1()
        config = {"some_config": {"param1": initial}}
        action.init_dict(config)
        assert action.values[0] == 0.4
        assert config["some_config"]["param1"][0][0] == 0.4


def test_action_init_dict_rejects_value_not_at_t_initial():
    # A single value at a time other than t_initial must be rejected.
    action = CustomAction1()
    config = {"some_config": {"param1": {5: 0.4}}}
    with pytest.raises(RuntimeError, match="t_initial"):
        action.init_dict(config)


def test_action_init_dict_accepts_value_at_custom_t_initial():
    # With numerics.t_initial set, the single value must be given at that time.
    action = CustomAction1()
    config = {
        "numerics": {"t_initial": 5.0},
        "some_config": {"param1": {5: 0.4}},
    }
    action.init_dict(config)
    assert action.values[0] == 0.4
    assert config["some_config"]["param1"][0][5.0] == 0.4


def test_action_init_dict_uses_restart_time():
    # When restarting from a previous run, the episode starts at restart.time
    # and the single value must be given at that time, not numerics.t_initial.
    action = CustomAction1()
    config = {
        "numerics": {"t_initial": 0.0},
        "restart": {"do_restart": True, "time": 10.0},
        "some_config": {"param1": {10.0: 0.4}},
    }
    action.init_dict(config)
    assert action.values[0] == 0.4
    assert config["some_config"]["param1"][0][10.0] == 0.4

    # With do_restart False, numerics.t_initial applies again.
    action = CustomAction1()
    config = {
        "numerics": {"t_initial": 0.0},
        "restart": {"do_restart": False, "time": 10.0},
        "some_config": {"param1": {10.0: 0.4}},
    }
    with pytest.raises(RuntimeError, match="t=10.0"):
        action.init_dict(config)


def test_action_get_state_variables():
    # Test state_var attribute returns the expected dict for CustomAction
    action = CustomAction()
    # CustomAction defines state_var as dict with scalars
    expected = {"scalars": ["param1", "param2"]}
    assert action.state_var == expected


# ------------------------
# ActionHandler tests
# ------------------------


def test_action_handler_get_actions():
    # Test retrieval of actions from ActionHandler with unique config_mapping
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    actions = [a for a in handler.get_actions().values()]
    assert actions == [a1, a2]


def test_action_handler_update_actions():
    # Test updating values of multiple actions via ActionHandler
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    actions_dict = {"CustomAction1": np.array([1.0]), "CustomAction2": np.array([2.0])}
    handler.update_actions(actions_dict)
    assert a1.values == [1.0]
    assert a2.values == [2.0]


def test_action_handler_update_actions_invalid_length():
    # Test that update_actions raises ValueError for wrong action name
    a1 = CustomAction()
    handler = ActionHandler([a1])
    actions_dict = {"WrongActionName": np.array([1.0, 2.0])}
    with pytest.raises(ValueError):
        handler.update_actions(actions_dict)


def test_action_handler_duplicate_keys():
    # Test that ActionHandler raises ValueError for duplicate config keys
    class A1(Action):
        name = "A1"
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        default_ramp_rate = [None]
        config_mapping = {("a",): 0}

    class A2(Action):
        name = "A2"
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        default_ramp_rate = [None]
        config_mapping = {("a",): 0}

    with pytest.raises(ValueError):
        ActionHandler([A1(), A2()])


def test_action_handler_ip_vloop_exclusive():
    # Test that ActionHandler raises ValueError if both Ip and Vloop actions are present
    ip = IpAction()
    vloop = VloopAction()
    with pytest.raises(ValueError):
        ActionHandler([ip, vloop])


def test_action_handler_get_action_variables():
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    variables = handler.get_action_variables()
    assert "scalars" in variables
    assert "param1" in variables["scalars"]
    assert "param2" in variables["scalars"]


def test_action_handler_validate_restart_scalars():
    # Matching restart-file scalars pass; mismatches raise; unknown scalars
    # and unmapped parameters (e.g. ECRH location/width) are skipped.
    handler = ActionHandler([IpAction(), EcrhAction()])
    handler.get_actions()["Ip"]._set_values([3e6])
    handler.get_actions()["ECRH"]._set_values([5e6, 0.3, 0.1])

    # Exact match passes, as does a value within the relative tolerance
    handler.validate_restart_scalars({"Ip": 3e6, "P_ecrh_e": 5e6})
    handler.validate_restart_scalars({"Ip": 3.001e6, "P_ecrh_e": 5e6})

    # Scalars missing from the file are skipped with a warning
    handler.validate_restart_scalars({"Ip": 3e6})

    # A mismatch raises
    with pytest.raises(ValueError, match="Restart consistency"):
        handler.validate_restart_scalars({"Ip": 10.5e6, "P_ecrh_e": 5e6})
    with pytest.raises(ValueError, match="P_ecrh_e"):
        handler.validate_restart_scalars({"Ip": 3e6, "P_ecrh_e": 20e6})


def test_action_handler_validate_restart_scalars_warns_on_missing(caplog):
    # A scalar absent from the restart file is skipped with a warning, not
    # silently ignored.
    handler = ActionHandler([IpAction()])
    handler.get_actions()["Ip"]._set_values([3e6])
    with caplog.at_level(logging.WARNING):
        handler.validate_restart_scalars({})
    assert "skipped" in caplog.text
    assert "Ip" in caplog.text


def test_action_handler_validate_restart_scalars_tolerance():
    # The comparison uses a 1% relative tolerance (file scalars are
    # post-processed floats).
    handler = ActionHandler([IpAction()])
    handler.get_actions()["Ip"]._set_values([1e6])
    handler.validate_restart_scalars({"Ip": 1.009e6})  # 0.9% off: passes
    with pytest.raises(ValueError, match="Restart consistency"):
        handler.validate_restart_scalars({"Ip": 1.011e6})  # 1.1% off: raises


def test_action_handler_validate_restart_scalars_unlisted_action():
    # Escape hatch: an action that lists no output scalar in state_var is
    # never checked (e.g. one deviating from the primary-parameter
    # convention).
    class UncheckedAction(Action):
        name = "Unchecked"
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        default_ramp_rate = [None]
        config_mapping = {("some_config", "p"): (0, 1)}
        state_var = {}

    handler = ActionHandler([UncheckedAction()])
    handler.validate_restart_scalars({"anything": 123.0})


def test_action_handler_build_action_space():
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    space = handler.build_action_space()
    assert set(space.spaces.keys()) == {"CustomAction1", "CustomAction2"}
    assert np.allclose(space.spaces["CustomAction1"].low, a1.min)
    assert np.allclose(space.spaces["CustomAction1"].high, a1.max)
    assert np.allclose(space.spaces["CustomAction2"].low, a2.min)
    assert np.allclose(space.spaces["CustomAction2"].high, a2.max)


# ------------------------
# Pre-configured Action classes
# ------------------------


def test_ip_action():
    # Test IpAction class attributes and config mapping
    ip = IpAction()
    assert ip.dimension == 1
    assert ip.min == [_MIN_IP_AMPS]
    assert ip.max == [np.inf]
    assert list(ip.get_mapping().keys())[0] == ("profile_conditions", "Ip")
    assert isinstance(ip.state_var, dict)
    assert ip.state_var == {"scalars": ["Ip"]}


def test_vloop_action():
    # Test VloopAction class attributes and config mapping
    vloop = VloopAction()
    assert vloop.dimension == 1
    assert vloop.min == [0.0]
    assert vloop.max == [np.inf]
    assert list(vloop.get_mapping().keys())[0] == ("profile_conditions", "v_loop_lcfs")
    assert isinstance(vloop.state_var, dict)
    assert vloop.state_var == {"scalars": ["v_loop_lcfs"]}


def test_ecrh_action():
    # Test EcrhAction class attributes and config mapping
    ecrh = EcrhAction()
    assert ecrh.dimension == 3
    np.testing.assert_array_equal(ecrh.min, [0.0, 0.0, 0.01])
    np.testing.assert_array_equal(ecrh.max, [np.inf, 1.0, np.inf])
    mapping = ecrh.get_mapping()
    assert ("sources", "ecrh", "P_total") in mapping
    assert ("sources", "ecrh", "gaussian_location") in mapping
    assert ("sources", "ecrh", "gaussian_width") in mapping
    assert isinstance(ecrh.state_var, dict)
    assert ecrh.state_var == {"scalars": ["P_ecrh_e"]}


def test_nbi_action():
    # Test NbiAction class attributes and config mapping
    nbi = NbiAction()
    assert nbi.dimension == 3
    np.testing.assert_array_equal(nbi.min, [0.0, 0.0, 0.01])
    np.testing.assert_array_equal(nbi.max, [np.inf, 1.0, np.inf])
    mapping = nbi.get_mapping()
    assert ("sources", "generic_heat", "P_total") in mapping
    assert ("sources", "generic_current", "I_generic") in mapping
    assert ("sources", "generic_heat", "gaussian_location") in mapping
    assert ("sources", "generic_heat", "gaussian_width") in mapping
    assert ("sources", "generic_current", "gaussian_location") in mapping
    assert ("sources", "generic_current", "gaussian_width") in mapping
    assert isinstance(nbi.state_var, dict)
    assert nbi.state_var == {"scalars": ["P_aux_generic_total"]}


def test_gas_puff_action():
    # Test GasPuffAction class attributes and config mapping
    puff = GasPuffAction()
    assert puff.dimension == 2
    np.testing.assert_array_equal(puff.min, [0.0, 0.01])
    np.testing.assert_array_equal(puff.max, [np.inf, np.inf])
    mapping = puff.get_mapping()
    assert ("sources", "gas_puff", "S_total") in mapping
    assert ("sources", "gas_puff", "puff_decay_length") in mapping
    assert isinstance(puff.state_var, dict)
    assert puff.state_var == {"scalars": ["S_gas_puff"]}


def test_gas_puff_action_init_dict_and_update():
    # Test init_dict and get_provider_updates for GasPuffAction
    puff = GasPuffAction()
    puff._set_values([1e22, 0.3])
    config = {"sources": {"gas_puff": {"S_total": None, "puff_decay_length": None}}}
    puff.init_dict(config)
    assert config["sources"]["gas_puff"]["S_total"][0][0] == 1e22
    assert config["sources"]["gas_puff"]["puff_decay_length"][0][0] == 0.3
    # Update at time=2.0: ramps from previous values (window start) to the
    # new values (window end).
    puff._set_values([2e22, 0.2])
    updates = puff.get_provider_updates(time=2.0, delta_t_a=1.0)
    assert updates["sources.gas_puff.S_total"].time.tolist() == [2.0, 3.0]
    assert updates["sources.gas_puff.S_total"].value.tolist() == [1e22, 2e22]
    assert updates["sources.gas_puff.puff_decay_length"].value.tolist() == [0.3, 0.2]


def test_ecrh_action_init_dict_and_update():
    # Test init_dict and get_provider_updates for EcrhAction
    ecrh = EcrhAction()
    ecrh._set_values([5e6, 0.3, 0.1])
    config = {
        "sources": {
            "ecrh": {"P_total": None, "gaussian_location": None, "gaussian_width": None}
        }
    }
    ecrh.init_dict(config)
    assert config["sources"]["ecrh"]["P_total"][0][0] == 5e6
    assert config["sources"]["ecrh"]["gaussian_location"][0][0] == 0.3
    assert config["sources"]["ecrh"]["gaussian_width"][0][0] == 0.1
    # Update at time=2.0: ramps from previous values (window start) to the
    # new values (window end).
    ecrh._set_values([6e6, 0.4, 0.2])
    updates = ecrh.get_provider_updates(time=2.0, delta_t_a=1.0)
    assert updates["sources.ecrh.P_total"].time.tolist() == [2.0, 3.0]
    assert updates["sources.ecrh.P_total"].value.tolist() == [5e6, 6e6]
    assert updates["sources.ecrh.gaussian_location"].value.tolist() == [0.3, 0.4]
    assert updates["sources.ecrh.gaussian_width"].value.tolist() == [0.1, 0.2]


def test_nbi_action_init_dict_and_update():
    # Test init_dict and get_provider_updates for NbiAction
    nbi = NbiAction()
    nbi._set_values(
        [10e6, 0.4, 0.2]
    )  # Updated for 3 dimensions: power, location, width
    config = {
        "sources": {
            "generic_heat": {
                "P_total": None,
                "gaussian_location": None,
                "gaussian_width": None,
            },
            "generic_current": {
                "I_generic": None,
                "gaussian_location": None,
                "gaussian_width": None,
            },
        }
    }
    nbi.init_dict(config)
    assert config["sources"]["generic_heat"]["P_total"][0][0] == 10e6
    # Current drive power should be calculated from heat power using nbi_w_to_ma factor
    expected_current = 10e6 * nbi.nbi_w_to_ma  # Default factor is 1/16e6
    assert config["sources"]["generic_current"]["I_generic"][0][0] == expected_current
    assert config["sources"]["generic_heat"]["gaussian_location"][0][0] == 0.4
    assert config["sources"]["generic_heat"]["gaussian_width"][0][0] == 0.2
    assert config["sources"]["generic_current"]["gaussian_location"][0][0] == 0.4
    assert config["sources"]["generic_current"]["gaussian_width"][0][0] == 0.2
    # Update at time=3.0: ramps from previous values (window start) to the
    # new values (window end).
    nbi._set_values([11e6, 0.5, 0.3])  # Updated for 3 dimensions
    updates = nbi.get_provider_updates(time=3.0, delta_t_a=1.0)
    assert updates["sources.generic_heat.P_total"].time.tolist() == [3.0, 4.0]
    assert updates["sources.generic_heat.P_total"].value.tolist() == [10e6, 11e6]
    assert updates["sources.generic_current.I_generic"].value.tolist() == [
        10e6 * nbi.nbi_w_to_ma,
        11e6 * nbi.nbi_w_to_ma,
    ]
    assert updates["sources.generic_heat.gaussian_location"].value.tolist() == [
        0.4,
        0.5,
    ]
    assert updates["sources.generic_heat.gaussian_width"].value.tolist() == [0.2, 0.3]
    assert updates["sources.generic_current.gaussian_location"].value.tolist() == [
        0.4,
        0.5,
    ]
    assert updates["sources.generic_current.gaussian_width"].value.tolist() == [
        0.2,
        0.3,
    ]
