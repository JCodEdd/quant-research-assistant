import pandas as pd
import pytest

from features.indicators.returns import Returns
from features.indicators.rolling_high import RollingHigh
from features.indicators.rolling_low import RollingLow


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        'High': [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        'Low': [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
    }
    return pd.DataFrame(data, index=dates)


# ============================================================================
# Standard Scenarios
# ============================================================================

def test_returns_computes_pct_change_accurately(sample_ohlcv_df):
    """Verifies that Returns computes exact percentage changes over the specified window."""
    # Arrange
    ret = Returns(window=1)

    # Act
    result_df = ret.compute(sample_ohlcv_df)
    result = result_df[ret.name]

    # Assert
    assert ret.name == "Returns_1"
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(11.5 / 10.5 - 1.0)
    assert result.iloc[9] == pytest.approx(19.5 / 18.5 - 1.0)


def test_rolling_high_finds_max_value_in_window(sample_ohlcv_df):
    """Verifies that RollingHigh finds the correct maximum High in the rolling window."""
    # Arrange
    rh = RollingHigh(window=3)

    # Act
    result_df = rh.compute(sample_ohlcv_df)
    result = result_df[rh.name]

    # Assert
    assert rh.name == "RollingHigh_3"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 13.0       # Max of [11.0, 12.0, 13.0]
    assert result.iloc[9] == 20.0       # Max of [18.0, 19.0, 20.0]


def test_rolling_low_finds_min_value_in_window(sample_ohlcv_df):
    """Verifies that RollingLow finds the correct minimum Low in the rolling window."""
    # Arrange
    rl = RollingLow(window=3)

    # Act
    result_df = rl.compute(sample_ohlcv_df)
    result = result_df[rl.name]

    # Assert
    assert rl.name == "RollingLow_3"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 9.0        # Min of [9.0, 10.0, 11.0]
    assert result.iloc[9] == 16.0       # Min of [16.0, 17.0, 18.0]


# ============================================================================
# Failure Scenarios
# ============================================================================

def test_rolling_high_raises_key_error_high_column_is_missing(sample_ohlcv_df):
    """Verifies that RollingHigh raises a KeyError if 'High' is absent."""
    # Arrange
    rh = RollingHigh(window=3)
    invalid_df = sample_ohlcv_df.drop(columns=['High'])

    # Act & Assert
    with pytest.raises(KeyError):
        rh.compute(invalid_df)


# ============================================================================
# Edge Cases
# ============================================================================

def test_returns_with_flat_prices_returns_zero(sample_ohlcv_df):
    """Verifies that Returns on completely flat prices yields exactly 0.0 returns."""
    # Arrange
    flat_df = sample_ohlcv_df.copy()
    flat_df['Close'] = 100.0
    ret = Returns(window=1)

    # Act
    result = ret.compute(flat_df)

    # Assert
    valid_values = result[ret.name].dropna()
    assert (valid_values == 0.0).all()