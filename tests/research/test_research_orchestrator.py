
from datetime import timezone, datetime

import pandas as pd
import pytest

from exceptions.data_exceptions import EmptyDatasetError
from features.indicators.rsi import RSI
from features.indicators.sma import SMA
from ingestion.cache import Cache
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from interfaces.data_provider import DataProvider
from models.market_data_request import MarketDataRequest
from research.research_orchestrator import ResearchOrchestrator


class InMemoryCache(Cache):
    """A lightweight in memory fake for cache"""

    def __init__(self):
        self._store = {}

    def is_month_cached(self, provider: str, ticker: str, year: int, month: int) -> bool:
        return (provider, ticker, year, month) in self._store

    def get_month(self, provider: str, ticker: str, year: int, month: int) -> pd.DataFrame | None:
        return self._store.get((provider, ticker, year, month))

    def save_month(
            self,
            provider: str,
            ticker: str,
            year: int,
            month: int,
            df: pd.DataFrame
    ) -> None:
        self._store[(provider, ticker, year, month)] = df.copy()

class FakeProvider(DataProvider):
    """A fake data provider that creates synthetic market data and tracks downloads"""

    def download_data(self, market_data_request: MarketDataRequest) -> pd.DataFrame:
        tickers = [market_data_request.tickers] if isinstance(market_data_request.tickers, str) else market_data_request.tickers
        dates = pd.date_range(
            market_data_request.start_date,
            market_data_request.end_date,
            tz=timezone.utc
        )

        if len(tickers) == 1:
            data = {
                'Open': [100.0 + i for i in range(len(dates))],
                'High': [105.0 + i for i in range(len(dates))],
                'Low': [95.0 + i for i in range(len(dates))],
                'Close': [102.0 + i for i in range(len(dates))],
                'Volume': [1000 + i * 10 for i in range(len(dates))]
            }
            return pd.DataFrame(data, index=dates)
        else:
            metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
            columns = pd.MultiIndex.from_product([metrics, tickers])
            data = [[100.0] * len(columns) for _ in range(len(dates))]
            return pd.DataFrame(data, index=dates, columns=columns)

class EmptyFakeProvider(DataProvider):
    """Fake provider simulating an asset with no data found"""

    def download_data(self, market_data_request: MarketDataRequest) -> pd.DataFrame:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

@pytest.fixture
def orchestrator():
    cache = InMemoryCache()
    provider = FakeProvider()
    validator = DataValidator()
    service = IngestionService(provider, cache, validator)
    return ResearchOrchestrator(service)


# ============================================================================
# Standard Scenarios
# ============================================================================

def test_research_orchestrator_fetches_enriched_data_with_warmup(orchestrator):
    """Verifies that ResearchOrchestrator fetches extended historical data for indicator warm-up and slices correctly"""
    # Arrange
    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 2, 10, tzinfo=timezone.utc)
    indicators = [SMA(window=5)]

    # Act
    result = orchestrator.get_enriched_data(
        provider="yahoo",
        tickers="AAPL",
        start_date=start_date,
        end_date=end_date,
        indicators=indicators
    )

    # Assert
    # 1. Output length matches requested range exactly (10 days: Feb 1 to Feb 10)
    assert len(result) == 10
    assert result.index[0] == start_date
    assert result.index[-1] == end_date

    # 2. Indicators are calculated and present, with NO NaNs at the start date
    # (because warm-up data was fetched prior to Feb 1)
    assert "SMA_5" in result.columns
    assert not pd.isna(result["SMA_5"]. iloc[0])


# ============================================================================
# Failure Scenarios
# ============================================================================

def test_research_orchestrator_propagates_empty_dataset_error(orchestrator):
    """Verifies that empty dataset errors from the underlying services are correctly propagated"""
    # Arrange
    # Swap provider to empty provider
    orchestrator.ingestion_service.data_provider = EmptyFakeProvider()
    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 2, 5, tzinfo=timezone.utc)

    # Act & Assert
    with pytest.raises(EmptyDatasetError):
        orchestrator.get_enriched_data(
            provider="yahoo",
            tickers="AAPL",
            start_date=start_date,
            end_date=end_date,
            indicators=[SMA(window=3)]
        )


# ============================================================================
# Edge Cases
# ============================================================================

def test_research_orchestrator_handles_multi_ticker_request(orchestrator):
    """Verifies that multi-ticker orchestration properly passes through and returns MultiIndex enriched data"""
    # Arrange
    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 2, 5, tzinfo=timezone.utc)
    indicators = [SMA(window=3), RSI(window=3)]

    # Act
    result = orchestrator.get_enriched_data(
        provider="yahoo",
        tickers=["AAPL", "MSFT"],
        start_date=start_date,
        end_date=end_date,
        indicators=indicators
    )

    # Assert
    assert isinstance(result.columns, pd.MultiIndex)
    assert "SMA_3" in result.columns.get_level_values(0)
    assert "RSI_3" in result.columns.get_level_values(0)
    assert "AAPL" in result.columns.get_level_values(1)
    assert "MSFT" in result.columns.get_level_values(1)
    assert len(result) == 5
