import pandas as pd

from features.indicator import Indicator


class RollingHigh(Indicator):
    """Rolling Maximum High over a window."""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"RollingHigh_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["High"].rolling(window=self.window).max()