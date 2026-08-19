import pandas as pd

from features.indicator import Indicator


class RollingLow(Indicator):
    """Rolling Minimum Low over a window"""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"RollingLow_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["Low"].rolling(window=self.window).min()