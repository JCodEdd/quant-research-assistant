import pandas as pd
import pytest

from features.feature_pipeline import FeaturePipeline
from features.indicator import Indicator


class DummyIndicator(Indicator):
    """Dummy indicator to isolate the pipeline from specific indicators logic"""

    def __init__(self, window: int) -> None:
        self.window = window

    @property
    def required_lookback(self) -> int:
        return self.window

    @property
    def name(self) -> str:
        return f"DummyIndicator_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df['Close']
        if isinstance(result, pd.Series):
            return result.to_frame(self.name)
        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result


@pytest.fixture
def sample_market_df():
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        'Close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
    }
    return pd.DataFrame(data, index=dates)


# ============================================================================
# Standard Scenarios
# ============================================================================
def test_feature_pipeline_applies_miltiple_indicators_successfully(sample_market_df):
    """Verifies that FeaturePipeline successfully computes and appends multiple indicators."""
    # Arrange
    pipeline = FeaturePipeline([DummyIndicator(3), DummyIndicator(1)])

    # Act
    enriched_df = pipeline.run(sample_market_df)

    # Assert
    assert "DummyIndicator_3" in enriched_df.columns
    assert "DummyIndicator_1" in enriched_df.columns
    assert len(enriched_df.columns) == len(sample_market_df.columns) + 2


def test_feature_pipeline_does_not_mutate_original_dataframe(sample_market_df):
    """Verifies that running FeaturePipeline leaves the input DataFrame unmodified (immutability)"""
    # Arrange
    pipeline = FeaturePipeline([DummyIndicator(3)])
    original_columns_count = len(sample_market_df.columns)

    # Act
    _ = pipeline.run(sample_market_df)

    # Assert
    assert len(sample_market_df.columns) == original_columns_count

def test_feature_pipeline_calculates_max_lookback():
    """Verifies that FeaturePipeline correctly calculates the maximum required lookback across indicators."""
    # Arrange
    # DummyIndicator lookback equals its window
    pipeline = FeaturePipeline([DummyIndicator(5), DummyIndicator(20), DummyIndicator(10)])

    # Act & Assert
    assert pipeline.get_max_lookback() == 20

# ============================================================================
# Failure Scenarios
# ============================================================================

def test_feature_pipeline_propagates_key_error_when_indicator_fails(sample_market_df):
    """Verifies that if any indicator in the pipeline raises KeyError, the pipeline propagates it"""
    # Arrange
    pipeline =FeaturePipeline([DummyIndicator(3)])
    invalid_df = sample_market_df.drop(columns=['Close'])

    # Act & Assert
    with pytest.raises(KeyError):
        pipeline.run(invalid_df)

# ============================================================================
# Edge Cases
# ============================================================================

def test_feature_pipeline_with_empty_indicator_list_returns_unmodified_dataframe(sample_market_df):
    """Verifies that an empty feature pipeline returns a copy of the original dataframe unaltered"""
    # Arrange
    pipeline = FeaturePipeline([])

    # Act
    enriched_df = pipeline.run(sample_market_df)

    # Assert
    pd.testing.assert_frame_equal(enriched_df, sample_market_df)


def test_feature_pipeline_empty_indicators_max_lookback_is_zero():
    """Verifies that an empty pipeline returns a max lookback of zero"""
    # Arrange
    pipeline = FeaturePipeline([])

    # Act and Assert
    assert pipeline.get_max_lookback() == 0