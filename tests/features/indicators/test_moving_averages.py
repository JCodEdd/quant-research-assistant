import pandas as pd
import pytest

from features.indicators.ema import EMA
from features.indicators.sma import SMA


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

def test_sma_computes_correct_rolling_mean(sample_market_df):
    """Verifies that SMA calculates the exact mathematical rolling mean for a given window."""
    # Arrange
    sma = SMA(window=3)

    # Act
    result = sma.compute(sample_market_df)

    # Assert
    assert sma.name == "SMA_3"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx((10.0 + 11.0 + 12.0) / 3)
    assert result.iloc[9] == pytest.approx((17.0 + 18.0 + 19.0) / 3)


def test_ema_computes_exponential_weighted_average(sample_market_df):
    """Verifies that EMA computes a valid exponentially weighted moving average series."""
    # Arrange
    # window=3 => alpha=0.5. Sequence: [10, 11, 12, 13...]
    # EMA_0 = 10.0
    # EMA_1 = 0.5 * 11 + 0.5 * 10.0 = 10.5
    # EMA_2 = 0.5 * 12 + 0.5 * 10.5 = 11.25
    ema = EMA(window=3)

    # Act
    result = ema.compute(sample_market_df)

    # Assert
    assert ema.name == "EMA_3"
    assert result.iloc[0] == 10.0
    assert result.iloc[1] == pytest.approx(10.5)
    assert result.iloc[2] == pytest.approx(11.25)

    assert not result.isna().all()
    # First value should equal the first close price
    assert result.iloc[0] == 10.0


# ============================================================================
# Failure Scenarios
# ============================================================================

def test_sma_raises_key_error_when_close_column_is_missing(sample_market_df):
    """Verifies that SMA raises a KeyError if the required 'Close' column is missing"""
    # Arrange
    sma = SMA(window=3)
    invalid_df = sample_market_df.drop(columns=['Close'])

    # Act & Assert
    with pytest.raises(KeyError):
        sma.compute(invalid_df)

def test_ema_raises_key_error_when_close_colum_is_missing(sample_market_df):
    """Verifies that EMA raises a KeyError if the required 'Close' column is missing."""
    # Arrange
    ema = EMA(window=3)
    invalid_df = sample_market_df.drop(columns=['Close'])

    # Act & Assert
    with pytest.raises(KeyError):
        ema.compute(invalid_df)


# ============================================================================
# Edge Cases
# ============================================================================

def test_sma_with_window_exceeding_dataset_length_returns_all_nans(sample_market_df):
    """Verifies that if window size is larger than the dataset length, SMA returns all NaNs."""
    # Arrange
    sma = SMA(window=15)

    # Act
    result = sma.compute(sample_market_df)

    # Assert
    assert result.isna().all()


def test_sma_with_constant_prices_returns_constant_values(sample_market_df):
    """Verifies that running SMA on constant prices returns the exact constant price once warmed up"""
    # Arrange
    constant_df = sample_market_df.copy()
    constant_df['Close'] = 50.0
    sma = SMA(window=3)

    # Act
    result = sma.compute(constant_df)

    # Assert
    assert result.iloc[2] == 50.0
    assert result.iloc[9] == 50.0
