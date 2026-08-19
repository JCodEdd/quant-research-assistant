import pandas as pd

from features.indicator import Indicator


class SMA(Indicator):
    """Simple Moving Average (SMA)."""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"SMA_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["Close"].rolling(window=self.window).mean()