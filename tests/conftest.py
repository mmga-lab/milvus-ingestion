"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def sample_data_rows():
    """Number of rows to use in test data generation."""
    return 10


@pytest.fixture(scope="session")
def test_seed():
    """Random seed for reproducible tests."""
    return 42
