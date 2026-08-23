import pandas as pd

from features.indicator import Indicator


class SMA(Indicator):
    """Simple Moving Average (SMA)."""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Only needs the an extra window of previous data"""
        return self.window

    @property
    def name(self) -> str:
        return f"SMA_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df["Close"].rolling(window=self.window).mean()
        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result