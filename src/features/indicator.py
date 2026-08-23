from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    """Abstract base class for technical indicators"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the indicator, used as column name when added to DataFrame."""

    @property
    @abstractmethod
    def required_lookback(self) -> int:
        """Number of periods required to compute the first valid indicator value."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the indicator values from the input OHLCV DataFrame

        Args:
            df (pd.DaraFrame): Input market data DataFrame with title-case columns (Open, High, Low, Close, Volume).

        Returns:
            pd.Series: Computed indicator series.
        """