"""Shared pytest fixtures."""

import pytest

from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings LRU cache before each test to avoid state bleed."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
