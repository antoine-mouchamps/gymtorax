"""Test suite for the ConfigLoader class.

This module contains comprehensive tests for the TORAX configuration loader, covering
normal operation, edge cases, and error conditions.
"""

from typing import Any

import pytest

from gymtorax.action_handler import Action, ActionHandler
from gymtorax.torax_wrapper.config_loader import ConfigLoader


# Simple action for ConfigLoader testing
class DummyAction(Action):
    name = "DummyAction"
    dimension = 1
    default_min = [0.0]
    default_max = [1.0]
    config_mapping = {}  # Empty mapping to avoid config conflicts
    state_var = {"scalars": ["test_var"]}


class TestConfigLoader:
    """Test cases for the ConfigLoader class."""

    @pytest.fixture
    def action_handler(self) -> ActionHandler:
        """Provide a test ActionHandler for ConfigLoader testing."""
        test_action = DummyAction()
        return ActionHandler([test_action])

    @pytest.fixture
    def valid_config(self) -> dict[str, Any]:
        """Provide a valid TORAX configuration for testing.

        This is a simplified version of the full TORAX config that contains the
        essential fields needed for testing the ConfigLoader functionality.
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

    def test_update_config(self, valid_config, action_handler):
        """Test updating the config with an action."""
        loader = ConfigLoader(valid_config, action_handler)
        action = {"DummyAction": [0.5]}
        loader.update_config(action, current_time=10.0, delta_t_a=2.0)
        # Check that t_initial and t_final are updated
        assert loader.config_dict["numerics"]["t_initial"] == 10.0
        assert loader.config_dict["numerics"]["t_final"] == 12.0

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
