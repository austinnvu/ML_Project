"""
Tests for backtesting simulator.
"""

import pytest
from src.backtest.simulator import run_backtest


class TestSimulator:
    """Tests for simulator module."""

    def test_imports(self):
        """Test that simulator imports correctly."""
        assert run_backtest is not None
