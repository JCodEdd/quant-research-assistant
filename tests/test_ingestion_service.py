from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ingestion.cache import Cache
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from interfaces.data_provider import DataProvider
from models.market_data_request import MarketDataRequest


class FakeProvider(DataProvider):
    
    def download_data(
            self, 
            market_data_request: MarketDataRequest
        ) -> pd.DataFrame:
        # Return a dummy DataFrame for testing purposes

        dates = pd.date_range(
            "2024-1-1",
            periods=5
        )

        data = {
            'open': [1,2,3,4,5],
            'high': [2,3,4,5,6],
            'low': [0,1,2,3,4],
            'close': [1,2,3,4,5],
            'volume': [100, 200, 300, 400, 500]
        }
        
        return pd.DataFrame(data, index=dates)

@pytest.fixture
def market_request():
    return MarketDataRequest(
        provider="yahoo",
        ticker="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 5, tzinfo=timezone.utc)
    )

@pytest.fixture
def mock_cache():
    return MagicMock(spec=Cache)  # Using MagicMock to simulate the cache behavior

@pytest.fixture
def mock_validator():
    return MagicMock(spec=DataValidator)  # Using MagicMock to simulate the validator behavior

@pytest.fixture
def fake_provider():
    return FakeProvider()  # Using the FakeProvider for testing

def test_get_data_cache_hint(market_request, mock_cache, mock_validator, fake_provider):
    # Setup
    cached_df = fake_provider.download_data(market_request) 
    mock_cache.exists_in_cache.return_value = True
    mock_cache.load_from_cache.return_value = cached_df

    service = IngestionService(
        data_provider=fake_provider,
        cache=mock_cache,
        validator=mock_validator
    )

    # Excecute
    result = service.get_data(market_request)

    # Assert
    mock_cache.exists_in_cache.assert_called_once_with(market_request)
    mock_cache.load_from_cache.assert_called_once_with(market_request)
    pd.testing.assert_frame_equal(result, cached_df)

def test_get_get_data_cache_miss(market_request, mock_cache, mock_validator):
    # Setup
    provider = MagicMock(spec=DataProvider)
    downloaded_df = FakeProvider().download_data(market_request)
    provider.download_data.return_value = downloaded_df

    mock_cache.exists_in_cache.return_value = False
    service = IngestionService(
        data_provider=provider,
        cache=mock_cache,
        validator=mock_validator
    )
  
    # Excecute
    result = service.get_data(market_request)

    # Assert
    mock_cache.exists_in_cache.assert_called_once_with(market_request)
    provider.download_data.assert_called_once_with(market_request)
    mock_validator.validate_data.assert_called_once_with(downloaded_df)
    mock_cache.load_from_cache.assert_not_called()
    pd.testing.assert_frame_equal(result, downloaded_df)