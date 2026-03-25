"""
Tests for feature engineering module.
"""

import pytest
from src.features.feature_builder import build_features


class TestFeatureBuilder:
    """Tests for feature_builder module."""

    def test_imports(self):
        """Test that feature_builder imports correctly."""
        assert build_features is not None
