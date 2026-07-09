from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gymtorax.torax_wrapper import torax_app
from gymtorax.torax_wrapper.torax_app import ToraxApp


# Dummy ConfigLoader for testing
class DummyConfigLoader:
    def __init__(self):
        self.config_torax = MagicMock()
        self.config_torax.geometry.build_provider = MagicMock()
        self.config_torax.numerics.t_initial = 0.0
        self.config_torax.restart = MagicMock()
        self.config_torax.restart.do_restart = False
        self.apply_action = MagicMock()

    def get_total_simulation_time(self):
        return 1.0

    def set_total_simulation_time(self, t):
        pass

    def get_initial_simulation_time(self, restart=False):
        return 0.0


@pytest.fixture
def torax_app_fixture():
    with (
        patch("gymtorax.torax_wrapper.torax_app.RuntimeParamsProvider"),
        patch("gymtorax.torax_wrapper.torax_app.make_step_fn"),
        patch("gymtorax.torax_wrapper.torax_app.SimulationStepFn") as MockStepFn,
        patch(
            "gymtorax.torax_wrapper.torax_app.get_initial_state_and_post_processed_outputs",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "gymtorax.torax_wrapper.torax_app.StateHistory"
        ) as MockStateHistory,
        patch("gymtorax.torax_wrapper.torax_app.DataTree"),
    ):
        step_instance = MockStepFn.return_value
        step_instance.jitted_fixed_time_step.return_value = (
            MagicMock(),
            MagicMock(),
        )
        step_instance.check_for_errors.return_value = torax_app.SimError.NO_ERROR
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = MagicMock()
        yield ToraxApp(DummyConfigLoader(), delta_t_a=0.1, store_history=True)


def test_init_sets_attributes(torax_app_fixture):
    """Test that __init__ sets up the main attributes."""
    app = torax_app_fixture
    assert app.initial_config is not None
    assert app.delta_t_a == 0.1
    assert app.is_started is False
    assert hasattr(app, "history_list")


def test_reset_initializes_simulation(torax_app_fixture):
    """Test the reset method initializes the simulation and sets is_started to True."""
    app = torax_app_fixture
    app.reset()
    assert app.is_started is True
    assert app.state is not None
    assert app.current_sim_state is not None or app.current_sim_output is not None


def test_update_config_updates_config(torax_app_fixture):
    """Test update_config calls config.apply_action and updates providers."""
    app = torax_app_fixture
    app.reset()
    action = np.array([1.0, 2.0])
    app.update_config(action)
    app.config.apply_action.assert_called_once_with(
        action, app.t_current, app.delta_t_a
    )
    assert app.step_fn is not None


def test_run_successful(torax_app_fixture):
    """Test run executes a simulation step and returns (True, False) when not finished."""
    app = torax_app_fixture
    app.reset()
    app.t_current = 0.0
    app.t_final = 1.0
    result, done = app.run()
    assert result is True
    assert done == (app.t_current > app.t_final)


def test_run_without_reset_raises():
    """Test run raises RuntimeError if reset was not called."""
    app = ToraxApp(DummyConfigLoader(), delta_t_a=0.1)
    with pytest.raises(RuntimeError):
        app.run()


def test_save_output_file_calls_netcdf(torax_app_fixture, tmp_path):
    """Test save_output_file calls to_netcdf on the state history object."""
    app = torax_app_fixture
    app.reset()
    app.history_list = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
    with patch(
        "gymtorax.torax_wrapper.torax_app.StateHistory"
    ) as MockStateHistory:
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = MagicMock()
        instance.simulation_output_to_xr.return_value.to_netcdf = MagicMock()
        file_name = str(tmp_path / "testfile.nc")
        app.save_output_file(file_name)
        instance.simulation_output_to_xr.return_value.to_netcdf.assert_called()


def test_get_state_data_returns_data(torax_app_fixture):
    """Test get_state_data returns the data from state.simulation_output_to_xr."""
    app = torax_app_fixture
    app.reset()
    app.state.simulation_output_to_xr = MagicMock(return_value="data")
    data = app.get_state_data()
    assert data == "data"


def test_run_returns_false_on_sim_error():
    """Test run returns (False, False) if sim_error is not NO_ERROR."""
    with (
        patch("gymtorax.torax_wrapper.torax_app.RuntimeParamsProvider"),
        patch("gymtorax.torax_wrapper.torax_app.make_step_fn"),
        patch("gymtorax.torax_wrapper.torax_app.SimulationStepFn") as MockStepFn,
        patch(
            "gymtorax.torax_wrapper.torax_app.get_initial_state_and_post_processed_outputs",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "gymtorax.torax_wrapper.torax_app.StateHistory"
        ) as MockStateHistory,
        patch("gymtorax.torax_wrapper.torax_app.DataTree"),
    ):
        step_instance = MockStepFn.return_value
        step_instance.jitted_fixed_time_step.return_value = (
            MagicMock(),
            MagicMock(),
        )
        step_instance.check_for_errors.return_value = (
            torax_app.SimError.NAN_DETECTED
        )
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = MagicMock()
        app = ToraxApp(DummyConfigLoader(), delta_t_a=0.1, store_history=True)
        app.reset()
        app.t_current = 0.0
        app.t_final = 1.0
        result, done = app.run()
        assert result is False
        assert done is False


def test_update_config_raises_value_error(torax_app_fixture):
    """Test update_config raises ValueError if config.apply_action raises ValueError."""
    app = torax_app_fixture
    app.reset()

    def raise_value_error(*a, **kw):
        raise ValueError("fail")

    app.config.apply_action = raise_value_error
    with pytest.raises(ValueError):
        app.update_config(np.array([1.0]))


def test_save_output_file_raises_if_store_history_false():
    """Test save_output_file raises RuntimeError if store_history is False."""
    app = ToraxApp(DummyConfigLoader(), delta_t_a=0.1, store_history=False)
    with pytest.raises(RuntimeError):
        app.save_output_file("dummy.nc")


def test_save_output_file_raises_on_write_error(torax_app_fixture, tmp_path):
    """Test save_output_file raises ValueError if to_netcdf fails."""
    app = torax_app_fixture
    app.reset()
    app.history_list = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
    with patch(
        "gymtorax.torax_wrapper.torax_app.StateHistory"
    ) as MockStateHistory:
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = MagicMock()

        def raise_exc(*a, **kw):
            raise Exception("fail")

        instance.simulation_output_to_xr.return_value.to_netcdf = raise_exc
        file_name = str(tmp_path / "testfile.nc")
        with pytest.raises(ValueError):
            app.save_output_file(file_name)


def test_get_state_data_raises_if_state_none():
    """Test get_state_data raises RuntimeError if state is None."""
    app = ToraxApp(DummyConfigLoader(), delta_t_a=0.1)
    app.state = None
    with pytest.raises(RuntimeError):
        app.get_state_data()


def test_history_list_appends_on_reset(torax_app_fixture):
    """Test that history_list is re-initialized to length 1 after each reset."""
    app = torax_app_fixture
    app.reset()
    assert len(app.history_list) == 1
    app.reset()
    assert len(app.history_list) == 1


def test_reset_with_restart_true():
    """Test reset initializes from file if restart.do_restart is True."""
    dummy = DummyConfigLoader()
    dummy.config_torax.restart.do_restart = True
    with (
        patch("gymtorax.torax_wrapper.torax_app.RuntimeParamsProvider"),
        patch("gymtorax.torax_wrapper.torax_app.make_step_fn"),
        patch("gymtorax.torax_wrapper.torax_app.SimulationStepFn"),
        patch(
            "gymtorax.torax_wrapper.torax_app.get_initial_state_and_post_processed_outputs",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "gymtorax.torax_wrapper.torax_app.get_initial_state_and_post_processed_outputs_from_file",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_from_file,
        patch("gymtorax.torax_wrapper.torax_app.StateHistory") as MockStateHistory,
        patch("gymtorax.torax_wrapper.torax_app.DataTree"),
    ):
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = MagicMock()
        app = ToraxApp(dummy, delta_t_a=0.1, store_history=True)
        app.reset()
        mock_from_file.assert_called_once()


def test_get_output_datatree_returns_datatree(torax_app_fixture):
    """Test get_output_datatree returns a DataTree when store_history is True."""
    app = torax_app_fixture
    app.reset()
    app.history_list = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
    with patch(
        "gymtorax.torax_wrapper.torax_app.StateHistory"
    ) as MockStateHistory:
        instance = MockStateHistory.return_value
        instance.simulation_output_to_xr.return_value = "datatree"
        result = app.get_output_datatree()
        assert result == "datatree"


def test_get_output_datatree_raises_if_store_history_false():
    """Test get_output_datatree raises RuntimeError if store_history is False."""
    app = ToraxApp(DummyConfigLoader(), delta_t_a=0.1, store_history=False)
    with pytest.raises(RuntimeError):
        app.get_output_datatree()
