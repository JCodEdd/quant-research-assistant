import pandas as pd
import pytest

from features.indicators.atr import ATR
from features.indicators.ema import EMA
from features.indicators.returns import Returns
from features.indicators.rolling_high import RollingHigh
from features.indicators.rolling_low import RollingLow
from features.indicators.rsi import RSI
from features.indicators.sma import SMA
from features.indicators.volatility import RollingVolatility


@pytest.mark.parametrize(
    "indicator_cls,kwargs,expected_multiplier,expected_offset",
    [
        (SMA, {"window": 10}, 1, 0),
        (EMA, {"window": 15}, 3, 0),
        (RSI, {"window": 14}, 5, 0),
        (ATR, {"window": 14}, 5, 0),
        (RollingVolatility, {"window": 20}, 1, 1),
        (Returns, {"window": 1}, 1, 0),
        (RollingHigh, {"window": 5}, 1, 0),
        (RollingLow, {"window": 5}, 1, 0),
    ],
)
def test_indicator_required_lookback_property(indicator_cls, kwargs, expected_multiplier, expected_offset):
    """Verifies that all indicators correctly report their required_lookback property."""
    # Arrange
    indicator = indicator_cls(**kwargs)
    expected_lookback = (kwargs["window"] * expected_multiplier) + expected_offset

    # Act & Assert
    assert indicator.required_lookback == expected_lookback


def test_indicators_support_multiticker_dataframe():
    """Verifies that indicators compute correctly when given a multi-ticker DataFrame"""
    # Arrange
    dates = pd.date_range("2023-01-01", periods=10)

    # Create multi-ticker data (e.g., AAPL and MSFT columns for Open, High, Low, Close, Volume)
    # Using a simple DataFrame where Close has multiple columns
    data = {
        ("Close", "AAPL"): [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ("Close", "MSFT"): [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
        ("High", "AAPL"): [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        ("High", "MSFT"): [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0],
        ("Low", "AAPL"): [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ("Low", "MSFT"): [99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0],
    }
    multi_df = pd.DataFrame(data, index=dates)
    multi_df.columns = pd.MultiIndex.from_tuples(data.keys())

    # Test SMA
    sma = SMA(window=3)
    sma_result = sma.compute(multi_df)
    assert isinstance(sma_result, pd.DataFrame)
    # The columns should have a top-level of SMA_3
    assert isinstance(sma_result.columns, pd.MultiIndex)
    assert sma_result.columns.levels[0][0] == sma.name
    assert sma_result[sma.name].iloc[2]["AAPL"] == pytest.approx(11.0)

    # Test ATR
    atr = ATR(window=3)
    atr_result = atr.compute(multi_df)
    assert isinstance(atr_result, pd.DataFrame)
    assert isinstance(atr_result.columns, pd.MultiIndex)
    assert atr_result.columns.levels[0][0] == atr.name
    assert not atr_result[atr.name].isna().all().all()