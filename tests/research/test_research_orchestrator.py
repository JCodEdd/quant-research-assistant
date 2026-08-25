from datetime import timezone, datetime

import pandas as pd
import pytest

from exceptions.data_exceptions import EmptyDatasetError
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from research.research_orchestrator import ResearchOrchestrator
from tests.fakes import DummyIndicator, FakeProvider, InMemoryCache
from tests.ingestion.test_ingestion_service import EmptyFakeProvider


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
    indicators = [DummyIndicator(window=5)]

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
    assert "DummyIndicator_5" in result.columns
    assert not pd.isna(result["DummyIndicator_5"]. iloc[0])


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
            indicators=[DummyIndicator(window=3)]
        )


# ============================================================================
# Edge Cases
# ============================================================================

def test_research_orchestrator_handles_multi_ticker_request(orchestrator):
    """Verifies that multi-ticker orchestration properly passes through and returns MultiIndex enriched data"""
    # Arrange
    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 2, 5, tzinfo=timezone.utc)
    indicators = [DummyIndicator(window=3), DummyIndicator(window=3)]

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
    assert "DummyIndicator_3" in result.columns.get_level_values(0)
    assert "DummyIndicator_3" in result.columns.get_level_values(0)
    assert "AAPL" in result.columns.get_level_values(1)
    assert "MSFT" in result.columns.get_level_values(1)
    assert len(result) == 5
