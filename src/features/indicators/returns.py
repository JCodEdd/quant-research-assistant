import pandas as pd

from features.indicator import Indicator


class Returns(Indicator):
    """Percentage returns over a rolling window."""

    def __init__(self, window: int = 1):
        self.window = window

    @property
    def name(self) -> str:
        return f"Returns_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["Close"].pct_change(periods=self.window)