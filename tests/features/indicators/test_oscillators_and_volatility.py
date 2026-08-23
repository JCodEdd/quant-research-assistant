import pandas as pd
import pytest

from features.indicators.atr import ATR
from features.indicators.rsi import RSI
from features.indicators.volatility import RollingVolatility


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        'Open': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'High': [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        'Low': [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
        'Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    }
    return pd.DataFrame(data, index=dates)


# ============================================================================
# Standard Scenarios
# ============================================================================

def test_rsi_computes_valid_values(sample_ohlcv_df):
    """Verifies that RSI generates the exact mathematical value for a known sequence and values bounded strictly between 0 and 100."""
    # Arrange
    # Close: [10.5, 11.5, 12.5, 13.5...]
    # Diff starts at index 1: [NaN, 1.0, 1.0, 1.0...]
    # Window=3 => min_periods=3 valid diffs => valid from index 3
    # Index 3: Avg Gain=1.0, Avg Loss=0.0 -> RSI = 100.0
    rsi = RSI(window=3)

    # Act
    result_df = rsi.compute(sample_ohlcv_df)
    result = result_df[rsi.name]

    # Assert
    assert rsi.name == "RSI_3"
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == 100.0

    # Fairly trivial but still a safeguard
    valid_values = result.dropna()
    assert (valid_values >= 0.0).all()
    assert (valid_values <= 100.0).all()


def test_atr_computes_valid_values(sample_ohlcv_df):
    """Verifies that ATR generates the exact mathematical value for a known sequence and positive volatility values reflecting hig-low ranges"""
    # Arrange
    # TR: [2.0, 2.0, 2.0, 2.0...]
    # Window=3 => min_periods=3 => iloc[0], iloc[1] are NaN
    # Index 2: Exponential moving average of 2.0 = 2.0
    atr = ATR(window=3)

    # Act
    result_df = atr.compute(sample_ohlcv_df)
    result = result_df[atr.name]

    # Assert
    assert atr.name == "ATR_3"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2.0

    # Fairly trivial but still a safeguard
    valid_values = result.dropna()
    assert (valid_values > 0).all()


def test_volatility_computes_valid_values(sample_ohlcv_df):
    """Verifies that RollingVolatility computes the exact rolling standard deviation and non-negative standard deviation values."""
    # Arrange
    # Close: [10.5, 11.5, 12.5, 13.5...]
    # Returns: [NaN, 11.5/10.5-1, 12.5/11.5-1, 13.5/12.5-1...]
    # r1 = 0.095238, r2 = 0.086957, r3 = 0.080000
    # std([0.095238, 0.086957, 0.080000]) approx 0.00763
    vol = RollingVolatility(window=3)

    # Act
    result_df = vol.compute(sample_ohlcv_df)
    result = result_df[vol.name]

    # Assert
    assert vol.name == "Volatility_3"
    assert result.iloc[3] == pytest.approx(0.0076376, abs=1e-5)

    # Fairly trivial but still a safeguard
    valid_values = result.dropna()
    assert (valid_values >= 0).all()


# ============================================================================
# Failure Scenarios
# ============================================================================

def test_atr_raises_key_error_when_high_or_low_column_is_missing(sample_ohlcv_df):
    """Verifies that ATR raises a KeyError when essential price columns are absent."""
    # Arrange
    atr = ATR(window=3)
    invalid_df = sample_ohlcv_df.drop(columns=['High'])

    # Act & Assert
    with pytest.raises(KeyError):
        atr.compute(invalid_df)


# ============================================================================
# Edge Cases
# ============================================================================

def test_volatility_with_constant_prices_evaluates_to_zero(sample_ohlcv_df):
    """Verifies that RollingVolatility on a flat line of constant prices is 0.0"""
    # Arrange
    flat_df = sample_ohlcv_df.copy()
    flat_df['Close'] = 100.0
    vol = RollingVolatility(window=3)

    # Act
    result = vol.compute(flat_df)

    # Assert
    valid_values = result[vol.name].dropna()
    assert (valid_values == 0.0).all()
