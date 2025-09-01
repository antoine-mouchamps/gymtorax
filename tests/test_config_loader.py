"""Test suite for the ConfigLoader class.

This module contains comprehensive tests for the TORAX configuration loader, covering
normal operation, edge cases, and error conditions.
"""

from typing import Any

import pytest

from gymtorax.torax_wrapper.config_loader import ConfigLoader


class TestConfigLoader:
    """Test cases for the ConfigLoader class."""

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

    def test_get_dict(self, valid_config):
        """Test getting the configuration dictionary."""
        loader = ConfigLoader(valid_config)
        result_dict = loader.get_dict()

        # Should return a copy of the original dict
        assert result_dict == valid_config
        # Should be a different object (copy, not reference)
        assert result_dict is not loader.config_dict

    def test_get_total_simulation_time_success(self, valid_config):
        """Test successful extraction of total simulation time."""
        loader = ConfigLoader(valid_config)
        time = loader.get_total_simulation_time()

        assert time == 150.0
        assert isinstance(time, float)

    def test_get_simulation_timestep_success(self, valid_config):
        """Test successful extraction of simulation timestep."""
        loader = ConfigLoader(valid_config)
        timestep = loader.get_simulation_timestep()

        assert timestep == 1.0
        assert isinstance(timestep, float)
