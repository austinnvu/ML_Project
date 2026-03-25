"""
Tests for data fetching and preprocessing modules.
"""

import pytest
from src.data.nba_fetcher import fetch_games, fetch_team_stats


class TestNBAFetcher:
    """Tests for nba_fetcher module."""

    def test_imports(self):
        """Test that nba_fetcher imports correctly."""
        assert fetch_games is not None
        assert fetch_team_stats is not None


class TestPreprocess:
    """Tests for preprocess module."""

    def test_imports(self):
        """Test that preprocess module imports correctly."""
        from src.data.preprocess import merge_and_clean, save_processed
        assert merge_and_clean is not None
        assert save_processed is not None
