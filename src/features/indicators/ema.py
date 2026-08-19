import pandas as pd

from features.indicator import Indicator


class EMA(Indicator):
    """Exponential Moving Average (EMA)"""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"EMA_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["Close"].ewm(span=self.window, adjust=False).mean()
