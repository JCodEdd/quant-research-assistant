from datetime import datetime, timezone

import pandas as pd
import pytest

from exceptions.data_exceptions import EmptyDatasetError
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from models.market_data_request import MarketDataRequest
from tests.fakes import EmptyFakeProvider, FakeProvider, InMemoryCache


@pytest.fixture
def market_request():
    return MarketDataRequest(
        provider="yahoo",
        tickers="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 5, tzinfo=timezone.utc)
    )


# ============================================================================
# Standard Scenarios
# ============================================================================

def test_get_data_cache_hit(market_request):
    """Verifies that a cache miss triggers provider download, validation, cache storage, and subsequent hints serve from cache."""
    # Arrange
    cache = InMemoryCache()
    provider = FakeProvider()
    validator = DataValidator()
    service = IngestionService(provider, cache, validator)

    assert not cache.is_month_cached("yahoo", "AAPL", 2024, 1)

    # Act 1: Initial request (Cache Miss)
    first_result = service.get_data(market_request)

    # Assert 1: Data returned and cache populated
    assert not first_result.empty
    assert len(first_result) == 5
    assert cache.is_month_cached("yahoo", "AAPL", 2024, 1)
    assert provider.download_count == 1

    # Act 2: Repeat request (Cache Hit)
    second_result = service.get_data(market_request)

    # Assert 2: Result served directly from cache without hitting provider again
    pd.testing.assert_frame_equal(first_result, second_result)
    assert provider.download_count == 1     # No additional download occurred


# ============================================================================
# Failure Scenarios
# ============================================================================

def test_ingestion_service_raises_empty_dataset_error_when_provider_returns_no_data(market_request):
    # Arrange
    cache = InMemoryCache()
    provider = EmptyFakeProvider()
    validator = DataValidator()
    service = IngestionService(provider, cache, validator)

    # Act & Assert
    with pytest.raises(EmptyDatasetError):
        service.get_data(market_request)

    # Cache should remain empty
    assert not cache.is_month_cached("yahoo", "AAPL", 2024, 1)


# ============================================================================
# Edge Cases
# ============================================================================

def test_ingestion_service_for_multi_ticker_partial_cache_fetches_missing_only():
    """Verifies multi-ticker request fetches only missing ticker, combining with already cached data"""
    # Assert
    cache = InMemoryCache()
    provider = FakeProvider()
    validator = DataValidator()
    service = IngestionService(provider, cache, validator)

    # Pre-populate AAPL in the cache for 2024-01
    aapl_dates = pd.date_range("2024-01-01", "2024-01-31 23:59:59", tz=timezone.utc)
    aapl_df = pd.DataFrame({
        'Open': [150.0] * len(aapl_dates),
        'High': [155.0] * len(aapl_dates),
        'Low': [145.0] * len(aapl_dates),
        'Close': [152.0] * len(aapl_dates),
        'Volume': [5000] * len(aapl_dates)
    }, index=aapl_dates)
    cache.save_month("yahoo", "AAPL", 2024, 1, aapl_df)

    multi_request = MarketDataRequest(
        provider="yahoo",
        tickers=["AAPL", "MSFT"],
        start_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
    )

    # Act
    result = service.get_data(multi_request)

    # Assert
    # 1. Returned DataFrame is a multi-index panel with both AAPL and MSFT
    assert isinstance(result.columns, pd.MultiIndex)
    assert 'AAPL' in result.columns.get_level_values(1)
    assert 'MSFT' in result.columns.get_level_values(1)
    assert len(result) == 6     # 10th to 15th inclusive

    # 2. MSFT is now cached as well
    assert cache.is_month_cached("yahoo", "MSFT", 2024, 1)

    # 3. Only MSFT was download (1 call)
    assert provider.download_count == 1
