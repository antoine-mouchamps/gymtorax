"""Test suite for the ConfigLoader class.

This module contains comprehensive tests for the TORAX configuration loader, covering
normal operation, edge cases, and error conditions.
"""

from typing import Any

import pytest

from gymtorax.action_handler import (
    Action,
    ActionHandler,
    GenericHeatAction,
    IpAction,
    NbiAction,
)
from gymtorax.torax_wrapper.config_loader import ConfigLoader


# Simple action for ConfigLoader testing
class DummyAction(Action):
    name = "DummyAction"
    dimension = 1
    default_min = [0.0]
    default_max = [1.0]
    default_ramp_rate = [None]
    config_mapping = {}  # Empty mapping to avoid config conflicts
    state_var = {"scalars": ["test_var"]}


class TestConfigLoader:
    """Test cases for the `ConfigLoader` class."""

    @pytest.fixture
    def action_handler(self) -> ActionHandler:
        """Provide a test `ActionHandler` for `ConfigLoader` testing."""
        test_action = DummyAction()
        return ActionHandler([test_action])

    @pytest.fixture
    def valid_config(self) -> dict[str, Any]:
        """Provide a valid TORAX configuration for testing.

        This is a simplified version of the full TORAX config that contains the
        essential fields needed for testing the `ConfigLoader` functionality.
        """
        return {
            "profile_conditions": {},  # use default profile conditions
            "plasma_composition": {},  # use default plasma composition
            "numerics": {
                "t_final": 150.0,
                "fixed_dt": 1.0,
            },
            "geometry": {"geometry_type": "circular"},
            "neoclassical": {},
            "sources": {},
            "pedestal": {},
            "transport": {},
            "solver": {},
            "time_step_calculator": {},
        }

    def test_get_dict(self, valid_config, action_handler):
        """Test getting the configuration dictionary."""
        loader = ConfigLoader(valid_config, action_handler)
        result_dict = loader.get_dict()

        # Should return a copy of the original dict
        assert result_dict == valid_config
        # Should be a different object (copy, not reference)
        assert result_dict is not loader.config_dict

    def test_get_total_simulation_time_success(self, valid_config, action_handler):
        """Test successful extraction of total simulation time."""
        loader = ConfigLoader(valid_config, action_handler)
        time = loader.get_total_simulation_time()

        assert time == 150.0
        assert isinstance(time, float)

    def test_get_simulation_timestep_success(self, valid_config, action_handler):
        """Test successful extraction of simulation timestep."""
        loader = ConfigLoader(valid_config, action_handler)
        timestep = loader.get_simulation_timestep()

        assert timestep == 1.0
        assert isinstance(timestep, float)

    def test_set_total_simulation_time(self, valid_config, action_handler):
        """Test setting the total simulation time."""
        loader = ConfigLoader(valid_config, action_handler)
        loader.set_total_simulation_time(200.0)
        assert loader.get_total_simulation_time() == 200.0

    def test_get_initial_simulation_time_default(self, valid_config, action_handler):
        """Test getting initial simulation time when not set (should default to 0.0)."""
        loader = ConfigLoader(valid_config, action_handler)
        assert loader.get_initial_simulation_time() == 0.0

    def test_get_initial_simulation_time_explicit(self, valid_config, action_handler):
        """Test getting initial simulation time when explicitly set."""
        valid_config["numerics"]["t_initial"] = 5.0
        loader = ConfigLoader(valid_config, action_handler)
        assert loader.get_initial_simulation_time() == 5.0

    def test_get_n_grid_points_default(self, valid_config, action_handler):
        """Test getting n_grid_points when not set (should default to 25)."""
        loader = ConfigLoader(valid_config, action_handler)
        assert loader.get_n_grid_points() == 25

    def test_get_n_grid_points_explicit(self, valid_config, action_handler):
        """Test getting n_grid_points when explicitly set."""
        valid_config["geometry"]["n_rho"] = 42
        loader = ConfigLoader(valid_config, action_handler)
        assert loader.get_n_grid_points() == 42

    def test_apply_action(self, valid_config, action_handler):
        """Test applying an action and collecting provider updates."""
        loader = ConfigLoader(valid_config, action_handler)
        action = {"DummyAction": [0.5]}
        updates = loader.apply_action(action, current_time=10.0, delta_t_a=2.0)
        # DummyAction has an empty config_mapping, so no provider updates
        assert updates == {}
        # The action values were applied through the handler
        assert loader.get_current_action_values()["DummyAction"][0] == 0.5

    def test_ip_control_requires_ip_from_parameters(self, valid_config):
        """Test that Ip control with Ip_from_parameters=False fails at construction."""
        valid_config["geometry"]["Ip_from_parameters"] = False
        valid_config["profile_conditions"] = {"Ip": 15e6}
        handler = ActionHandler([IpAction()])
        with pytest.raises(ValueError, match="Ip_from_parameters"):
            ConfigLoader(valid_config, handler)

    def test_nbi_requires_use_absolute_current(self, valid_config):
        """NbiAction with a fraction-mode generic_current fails at construction.

        In fraction mode TORAX ignores the I_generic current written by the
        action (the external current follows fraction_of_total_current * Ip),
        so the combination must be rejected instead of silently doing the
        wrong thing.
        """
        valid_config["sources"] = {
            "generic_heat": {
                "P_total": 0.0,
                "gaussian_location": 0.25,
                "gaussian_width": 0.25,
            },
            "generic_current": {
                "I_generic": 0.0,
                "gaussian_location": 0.25,
                "gaussian_width": 0.25,
            },
        }
        with pytest.raises(ValueError, match="use_absolute_current"):
            ConfigLoader(valid_config, ActionHandler([NbiAction()]))

    def test_nbi_with_absolute_current_accepted(self, valid_config):
        """NbiAction with use_absolute_current=True constructs cleanly."""
        valid_config["sources"] = {
            "generic_heat": {
                "P_total": 0.0,
                "gaussian_location": 0.25,
                "gaussian_width": 0.25,
            },
            "generic_current": {
                "use_absolute_current": True,
                "I_generic": 0.0,
                "gaussian_location": 0.25,
                "gaussian_width": 0.25,
            },
        }
        ConfigLoader(valid_config, ActionHandler([NbiAction()]))

    def test_generic_heat_action_with_fraction_mode(self, valid_config):
        """Heating-only action with a prescribed fraction-of-Ip current is valid.

        The generic_current keys are not action-controlled here: the external
        current stays prescribed in the configuration (linked to Ip by TORAX),
        as in the TORAX iterhybrid_rampup example.
        """
        valid_config["sources"] = {
            "generic_heat": {
                "P_total": 0.0,
                "gaussian_location": 0.25,
                "gaussian_width": 0.25,
            },
            "generic_current": {
                "fraction_of_total_current": 0.15,
                "gaussian_location": 0.36,
                "gaussian_width": 0.075,
            },
        }
        ConfigLoader(valid_config, ActionHandler([GenericHeatAction()]))

    def test_validate_discretization_fixed(self, valid_config, action_handler):
        """Test validate_discretization for fixed type."""
        valid_config["time_step_calculator"]["calculator_type"] = "fixed"
        loader = ConfigLoader(valid_config, action_handler)
        loader.validate_discretization("fixed")  # Should not raise

    def test_validate_discretization_auto(self, valid_config, action_handler):
        """Test validate_discretization for auto type."""
        valid_config["time_step_calculator"]["calculator_type"] = "chi"
        loader = ConfigLoader(valid_config, action_handler)
        loader.validate_discretization("auto")  # Should not raise

    def test_validate_discretization_invalid(self, valid_config, action_handler):
        """Test validate_discretization with invalid type."""
        loader = ConfigLoader(valid_config, action_handler)
        with pytest.raises(ValueError):
            loader.validate_discretization("invalid_type")
