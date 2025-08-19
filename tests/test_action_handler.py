import pytest
import numpy as np

from gymtorax.action_handler import (
    Action, ActionHandler, IpAction, VloopAction, EcrhAction, NbiAction
)

# Dummy value for _MIN_IP_AMPS if not available
try:
    from torax._src.config.profile_conditions import _MIN_IP_AMPS
except ImportError:
    _MIN_IP_AMPS = 0

# ------------------------
# Custom Action for testing
# ------------------------
class CustomAction(Action):
    dimension = 2
    default_min = [0.0, -1.0]
    default_max = [10.0, 1.0]
    config_mapping = {
        ('some_config', 'param1'): 0,
        ('some_config', 'param2'): 1
    }

# ------------------------
# Action class tests
# ------------------------

def test_action_init_defaults():
    action = CustomAction()
    assert action.min == [0.0, -1.0]
    assert action.max == [10.0, 1.0]
    assert action.values == [0.0, -1.0]

def test_action_init_custom_bounds():
    action = CustomAction(min=[1.0, 0.0], max=[5.0, 0.5])
    assert action.min == [1.0, 0.0]
    assert action.max == [5.0, 0.5]
    assert action.values == [1.0, 0.0]

def test_action_invalid_dimension():
    class BadAction(Action):
        dimension = 0
        default_min = []
        default_max = []
        config_mapping = {}
    with pytest.raises(ValueError):
        BadAction()

def test_action_invalid_min_length():
    class BadAction(Action):
        dimension = 2
        default_min = [0.0]
        default_max = [1.0, 2.0]
        config_mapping = {('a',): 0, ('b',): 1}
    with pytest.raises(ValueError):
        BadAction()

def test_action_invalid_max_length():
    class BadAction(Action):
        dimension = 2
        default_min = [0.0, 1.0]
        default_max = [1.0]
        config_mapping = {('a',): 0, ('b',): 1}
    with pytest.raises(ValueError):
        BadAction()

def test_action_set_values_valid():
    action = CustomAction()
    action.set_values([5.0, 0.5])
    assert action.values == [5.0, 0.5]

def test_action_set_values_invalid_length():
    action = CustomAction()
    with pytest.raises(ValueError):
        action.set_values([1.0])

def test_action_repr():
    action = CustomAction()
    r = repr(action)
    assert "CustomAction" in r
    assert "values" in r
    assert "min" in r
    assert "max" in r

def test_action_get_mapping():
    action = CustomAction()
    mapping = action.get_mapping()
    assert isinstance(mapping, dict)
    assert ('some_config', 'param1') in mapping

def test_action_init_dict_and_update_to_config():
    action = CustomAction()
    action.set_values([2.0, 0.5])
    config = {'some_config': {'param1': None, 'param2': None}}
    action.init_dict(config)
    assert config['some_config']['param1'][0][0] == 2.0
    assert config['some_config']['param2'][0][0] == 0.5
    # Update at time=1.0
    action.set_values([3.0, 0.7])
    action.update_to_config(config, time=1.0)
    assert config['some_config']['param1'][0][1.0] == 3.0
    assert config['some_config']['param2'][0][1.0] == 0.7

def test_action_init_dict_keyerror():
    action = CustomAction()
    config = {}  # Missing structure
    with pytest.raises(KeyError):
        action.init_dict(config)

# ------------------------
# ActionHandler tests
# ------------------------

def test_action_handler_get_actions():
    a1 = CustomAction()
    a2 = CustomAction()
    handler = ActionHandler([a1, a2])
    actions = handler.get_actions()
    assert actions == [a1, a2]

def test_action_handler_update_actions():
    a1 = CustomAction()
    a2 = CustomAction()
    handler = ActionHandler([a1, a2])
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    handler.update_actions(arr)
    assert a1.values == [1.0, 2.0]
    assert a2.values == [3.0, 4.0]

def test_action_handler_update_actions_invalid_length():
    a1 = CustomAction()
    handler = ActionHandler([a1])
    arr = np.array([1.0])  # Should be length 2
    with pytest.raises(ValueError):
        handler.update_actions(arr)

def test_action_handler_duplicate_keys():
    class A1(Action):
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        config_mapping = {('a',): 0}
    class A2(Action):
        dimension = 1
        default_min = [0.0]
        default_max = [1.0]
        config_mapping = {('a',): 0}
    with pytest.raises(ValueError):
        ActionHandler([A1(), A2()])

def test_action_handler_ip_vloop_exclusive():
    ip = IpAction()
    vloop = VloopAction()
    with pytest.raises(ValueError):
        ActionHandler([ip, vloop])

# ------------------------
# Pre-configured Action classes
# ------------------------

def test_ip_action():
    ip = IpAction()
    assert ip.dimension == 1
    assert ip.min == [_MIN_IP_AMPS]
    assert ip.max == [np.inf]
    assert list(ip.get_mapping().keys())[0] == ('profile_conditions', 'Ip')

def test_vloop_action():
    vloop = VloopAction()
    assert vloop.dimension == 1
    assert vloop.min == [0.0]
    assert vloop.max == [np.inf]
    assert list(vloop.get_mapping().keys())[0] == ('profile_conditions', 'v_loop_lcfs')

def test_ecrh_action():
    ecrh = EcrhAction()
    assert ecrh.dimension == 3
    assert ecrh.min == [0.0, 0.0, 0.01]
    assert ecrh.max == [np.inf, 1.0, np.inf]
    mapping = ecrh.get_mapping()
    assert ('sources', 'ecrh', 'P_total') in mapping
    assert ('sources', 'ecrh', 'gaussian_location') in mapping
    assert ('sources', 'ecrh', 'gaussian_width') in mapping

def test_nbi_action():
    nbi = NbiAction()
    assert nbi.dimension == 4
    assert nbi.min == [0.0, 0.0, 0.0, 0.01]
    assert nbi.max == [np.inf, np.inf, 1.0, np.inf]
    mapping = nbi.get_mapping()
    assert ('sources', 'generic_heat', 'P_total') in mapping
    assert ('sources', 'generic_current', 'I_generic') in mapping
    assert ('sources', 'generic_heat', 'gaussian_location') in mapping
    assert ('sources', 'generic_heat', 'gaussian_width') in mapping
    assert ('sources', 'generic_current', 'gaussian_location') in mapping
    assert ('sources', 'generic_current', 'gaussian_width') in mapping

def test_ecrh_action_init_dict_and_update():
    ecrh = EcrhAction()
    ecrh.set_values([5e6, 0.3, 0.1])
    config = {'sources': {'ecrh': {'P_total': None, 'gaussian_location': None, 'gaussian_width': None}}}
    ecrh.init_dict(config)
    assert config['sources']['ecrh']['P_total'][0][0] == 5e6
    assert config['sources']['ecrh']['gaussian_location'][0][0] == 0.3
    assert config['sources']['ecrh']['gaussian_width'][0][0] == 0.1
    # Update at time=2.0
    ecrh.set_values([6e6, 0.4, 0.2])
    ecrh.update_to_config(config, time=2.0)
    assert config['sources']['ecrh']['P_total'][0][2.0] == 6e6
    assert config['sources']['ecrh']['gaussian_location'][0][2.0] == 0.4
    assert config['sources']['ecrh']['gaussian_width'][0][2.0] == 0.2

def test_nbi_action_init_dict_and_update():
    nbi = NbiAction()
    nbi.set_values([10e6, 2e6, 0.4, 0.2])
    config = {
        'sources': {
            'generic_heat': {'P_total': None, 'gaussian_location': None, 'gaussian_width': None},
            'generic_current': {'I_generic': None, 'gaussian_location': None, 'gaussian_width': None}
        }
    }
    nbi.init_dict(config)
    assert config['sources']['generic_heat']['P_total'][0][0] == 10e6
    assert config['sources']['generic_current']['I_generic'][0][0] == 2e6
    assert config['sources']['generic_heat']['gaussian_location'][0][0] == 0.4
    assert config['sources']['generic_heat']['gaussian_width'][0][0] == 0.2
    assert config['sources']['generic_current']['gaussian_location'][0][0] == 0.4
    assert config['sources']['generic_current']['gaussian_width'][0][0] == 0.2
    # Update at time=3.0
    nbi.set_values([11e6, 3e6, 0.5, 0.3])
    nbi.update_to_config(config, time=3.0)
    assert config['sources']['generic_heat']['P_total'][0][3.0] == 11e6
    assert config['sources']['generic_current']['I_generic'][0][3.0] == 3e6
    assert config['sources']['generic_heat']['gaussian_location'][0][3.0] == 0.5
    assert config['sources']['generic_heat']['gaussian_width'][0][3.0] == 0.3
    assert config['sources']['generic_current']['gaussian_location'][0][3.0] == 0.5
    assert config['sources']['generic_current']['gaussian_width'][0][3.0] == 0.3