import numpy as np
import pytest
from torax._src.config.profile_conditions import _MIN_IP_AMPS

from gymtorax.action_handler import (
    Action,
    ActionHandler,
    EcrhAction,
    IpAction,
    NbiAction,
    VloopAction,
)

# ------------------------
# Custom Actions for testing
# ------------------------


class CustomAction(Action):
    dimension = 2
    default_min = [0.0, -1.0]
    default_max = [10.0, 1.0]
    config_mapping = {("some_config", "param1"): 0, ("some_config", "param2"): 1}
    state_var = ()


class CustomAction1(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [1.0]
    config_mapping = {("some_config", "param1"): 0}
    state_var = ()


class CustomAction2(Action):
    dimension = 1
    default_min = [0.0]
    default_max = [1.0]
    config_mapping = {("some_config", "param2"): 0}
    state_var = ()


# ------------------------
# Action class tests
# ------------------------


def test_action_init_defaults():
    # Test default initialization of CustomAction
    action = CustomAction()
    assert action.min == [0.0, -1.0]
    assert action.max == [10.0, 1.0]
    assert action.values == [0.0, -1.0]


def test_action_init_custom_bounds():
    # Test CustomAction initialization with custom min/max bounds
    action = CustomAction(min=[1.0, 0.0], max=[5.0, 0.5])
    assert action.min == [1.0, 0.0]
    assert action.max == [5.0, 0.5]
    assert action.values == [1.0, 0.0]


def test_action_invalid_dimension():
    # Test that Action with invalid dimension raises ValueError
    class BadAction(Action):
        dimension = 0
        default_min = []
        default_max = []
        config_mapping = {}

    with pytest.raises(ValueError):
        BadAction()


def test_action_invalid_min_length():
    # Test that Action with invalid default_min length raises ValueError
    class BadAction(Action):
        dimension = 2
        default_min = [0.0]
        default_max = [1.0, 2.0]
        config_mapping = {("a",): 0, ("b",): 1}

    with pytest.raises(ValueError):
        BadAction()


def test_action_invalid_max_length():
    # Test that Action with invalid default_max length raises ValueError
    class BadAction(Action):
        dimension = 2
        default_min = [0.0, 1.0]
        default_max = [1.0]
        config_mapping = {("a",): 0, ("b",): 1}

    with pytest.raises(ValueError):
        BadAction()


def test_action_set_values_valid():
    # Test setting valid values for CustomAction
    action = CustomAction()
    action.set_values([5.0, 0.5])
    assert action.values == [5.0, 0.5]


def test_action_set_values_invalid_length():
    # Test that setting values with wrong length raises ValueError
    action = CustomAction()
    with pytest.raises(ValueError):
        action.set_values([1.0])


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


def test_action_init_dict_and_update_to_config():
    # Test init_dict and update_to_config methods for CustomAction
    action = CustomAction()
    action.set_values([2.0, 0.5])
    config = {"some_config": {"param1": None, "param2": None}}
    action.init_dict(config)
    assert config["some_config"]["param1"][0][0] == 2.0
    assert config["some_config"]["param2"][0][0] == 0.5
    # Update at time=1.0
    action.set_values([3.0, 0.7])
    action.update_to_config(config, time=1.0)
    assert config["some_config"]["param1"][0][1.0] == 3.0
    assert config["some_config"]["param2"][0][1.0] == 0.7


def test_action_init_dict_keyerror():
    # Test that init_dict raises KeyError for missing config structure
    action = CustomAction()
    config = {}  # Missing structure
    with pytest.raises(KeyError):
        action.init_dict(config)


def test_action_get_state_variables():
    # Test get_state_variables returns empty tuple for CustomAction
    action = CustomAction()
    # CustomAction does not define state_var, should be ()
    assert action.get_state_variables() == ()


# ------------------------
# ActionHandler tests
# ------------------------


def test_action_handler_get_actions():
    # Test retrieval of actions from ActionHandler with unique config_mapping
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    actions = handler.get_actions()
    assert actions == [a1, a2]


def test_action_handler_update_actions():
    # Test updating values of multiple actions via ActionHandler
    a1 = CustomAction1()
    a2 = CustomAction2()
    handler = ActionHandler([a1, a2])
    arr = np.array([1.0, 2.0])
    handler.update_actions(arr)
    assert a1.values == [1.0]
    assert a2.values == [2.0]


def test_action_handler_update_actions_invalid_length():
    # Test that update_actions raises ValueError for wrong array length
    a1 = CustomAction()
    handler = ActionHandler([a1])
    arr = np.array([1.0])  # Should be length 2
    with pytest.raises(ValueError):
        handler.update_actions(arr)


def test_action_handler_duplicate_keys():
    # Test that ActionHandler raises ValueError for duplicate config keys
    class A1(Action):
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        config_mapping = {("a",): 0}

    class A2(Action):
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        config_mapping = {("a",): 0}

    with pytest.raises(ValueError):
        ActionHandler([A1(), A2()])


def test_action_handler_ip_vloop_exclusive():
    # Test that ActionHandler raises ValueError if both Ip and Vloop actions are present
    ip = IpAction()
    vloop = VloopAction()
    with pytest.raises(ValueError):
        ActionHandler([ip, vloop])


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
    assert isinstance(ip.get_state_variables(), tuple)
    assert ip.get_state_variables() == (("scalars", "Ip"))


def test_vloop_action():
    # Test VloopAction class attributes and config mapping
    vloop = VloopAction()
    assert vloop.dimension == 1
    assert vloop.min == [0.0]
    assert vloop.max == [np.inf]
    assert list(vloop.get_mapping().keys())[0] == ("profile_conditions", "v_loop_lcfs")
    assert isinstance(vloop.get_state_variables(), tuple)
    assert vloop.get_state_variables() == (("scalars", "v_loop_lcfs"))


def test_ecrh_action():
    # Test EcrhAction class attributes and config mapping
    ecrh = EcrhAction()
    assert ecrh.dimension == 3
    assert ecrh.min == [0.0, 0.0, 0.01]
    assert ecrh.max == [np.inf, 1.0, np.inf]
    mapping = ecrh.get_mapping()
    assert ("sources", "ecrh", "P_total") in mapping
    assert ("sources", "ecrh", "gaussian_location") in mapping
    assert ("sources", "ecrh", "gaussian_width") in mapping
    assert isinstance(ecrh.get_state_variables(), tuple)
    assert ("scalars", "P_ecrh_e") in ecrh.get_state_variables()
    assert ("profiles", "p_ecrh_e") in ecrh.get_state_variables()


def test_nbi_action():
    # Test NbiAction class attributes and config mapping
    nbi = NbiAction()
    assert nbi.dimension == 4
    assert nbi.min == [0.0, 0.0, 0.0, 0.01]
    assert nbi.max == [np.inf, np.inf, 1.0, np.inf]
    mapping = nbi.get_mapping()
    assert ("sources", "generic_heat", "P_total") in mapping
    assert ("sources", "generic_current", "I_generic") in mapping
    assert ("sources", "generic_heat", "gaussian_location") in mapping
    assert ("sources", "generic_heat", "gaussian_width") in mapping
    assert ("sources", "generic_current", "gaussian_location") in mapping
    assert ("sources", "generic_current", "gaussian_width") in mapping
    assert isinstance(nbi.get_state_variables(), tuple)
    # Should contain both scalars and profiles keys
    assert any("scalars" in t for t in nbi.get_state_variables())
    assert any("profiles" in t for t in nbi.get_state_variables())


def test_ecrh_action_init_dict_and_update():
    # Test init_dict and update_to_config for EcrhAction
    ecrh = EcrhAction()
    ecrh.set_values([5e6, 0.3, 0.1])
    config = {
        "sources": {
            "ecrh": {"P_total": None, "gaussian_location": None, "gaussian_width": None}
        }
    }
    ecrh.init_dict(config)
    assert config["sources"]["ecrh"]["P_total"][0][0] == 5e6
    assert config["sources"]["ecrh"]["gaussian_location"][0][0] == 0.3
    assert config["sources"]["ecrh"]["gaussian_width"][0][0] == 0.1
    # Update at time=2.0
    ecrh.set_values([6e6, 0.4, 0.2])
    ecrh.update_to_config(config, time=2.0)
    assert config["sources"]["ecrh"]["P_total"][0][2.0] == 6e6
    assert config["sources"]["ecrh"]["gaussian_location"][0][2.0] == 0.4
    assert config["sources"]["ecrh"]["gaussian_width"][0][2.0] == 0.2


def test_nbi_action_init_dict_and_update():
    # Test init_dict and update_to_config for NbiAction
    nbi = NbiAction()
    nbi.set_values([10e6, 2e6, 0.4, 0.2])
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
    assert config["sources"]["generic_current"]["I_generic"][0][0] == 2e6
    assert config["sources"]["generic_heat"]["gaussian_location"][0][0] == 0.4
    assert config["sources"]["generic_heat"]["gaussian_width"][0][0] == 0.2
    assert config["sources"]["generic_current"]["gaussian_location"][0][0] == 0.4
    assert config["sources"]["generic_current"]["gaussian_width"][0][0] == 0.2
    # Update at time=3.0
    nbi.set_values([11e6, 3e6, 0.5, 0.3])
    nbi.update_to_config(config, time=3.0)
    assert config["sources"]["generic_heat"]["P_total"][0][3.0] == 11e6
    assert config["sources"]["generic_current"]["I_generic"][0][3.0] == 3e6
    assert config["sources"]["generic_heat"]["gaussian_location"][0][3.0] == 0.5
    assert config["sources"]["generic_heat"]["gaussian_width"][0][3.0] == 0.3
    assert config["sources"]["generic_current"]["gaussian_location"][0][3.0] == 0.5
    assert config["sources"]["generic_current"]["gaussian_width"][0][3.0] == 0.3
