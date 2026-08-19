import pandas as pd

from features.indicator import Indicator


class RollingVolatility(Indicator):
    """Rolling Volatility (Standard Deviation of Returns)"""

    def __init__(self, window: int = 20):
        self.window =  window

    @property
    def name(self) -> str:
        return f"Volatility_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["Close"].pct_change()
        return returns.rolling(window=self.window).std()