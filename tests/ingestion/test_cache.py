import pandas as pd
import pytest

from ingestion.cache import Cache


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=10)
    return pd.DataFrame({
        "Open": range(10, 20),
        "Close": range(11, 21)
    }, index=dates)

# ============================================================================
# Standard Scenarios
# ============================================================================

def test_cache_save_load_exists(tmp_path, sample_df):
    """Verifies that saving and then retrieving a monthly partition returns the exact same data."""
    # Arrange
    cache = Cache(tmp_path)
    provider = "yahoo"
    ticker = "AAPL"
    year = 2023
    month = 1

    # Initially, the cache should not have the data
    assert cache.is_month_cached(provider, ticker, year, month) is False
    assert cache.get_month(provider, ticker, year, month) is None

    # Act
    # Save the data to the cache
    cache.save_month(provider, ticker, year, month, sample_df)

    # Now, the cache should have the data
    assert cache.is_month_cached(provider, ticker, year, month) is True
    loaded_df = cache.get_month(provider, ticker, year, month)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(sample_df, loaded_df, check_freq=False)


# ============================================================================
# Failure / Miss Scenarios
# ============================================================================

def test_cache_get_month_when_not_cached_returns_none(tmp_path):
    """Verifies that requesting a month that has not been saved returns None."""
    # Arrange
    cache = Cache(tmp_path)
    provider = "yahoo"
    ticker = "AAPL"
    year = 2024
    month = 12

    # Act & Assert
    assert cache.is_month_cached(provider, ticker, year, month) is False
    assert cache.get_month(provider, ticker, year, month) is None

# ============================================================================
# Edge Cases
# ============================================================================

def test_cache_save_month_overwrites_existing_partition_correctly(tmp_path, sample_df):
    """Verifies that saving a partition for a month that already exists replaces the old data"""
    # Arrange
    cache = Cache(tmp_path)
    provider = "yahoo"
    ticker = "AAPL"
    year = 2023
    month = 5

    # Initial save
    cache.save_month(provider, ticker, year, month, sample_df)

    # Updated data
    updated_df = sample_df * 2.0

    # Act
    cache.save_month(provider, ticker, year, month, updated_df)

    # Assert
    loaded_df = cache.get_month(provider, ticker, year, month)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, updated_df, check_freq=False)