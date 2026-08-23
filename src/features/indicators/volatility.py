import pandas as pd

from features.indicator import Indicator


class RollingVolatility(Indicator):
    """Rolling Volatility (Standard Deviation of Returns)"""

    def __init__(self, window: int = 20):
        self.window =  window

    @property
    def required_lookback(self) -> int:
        """Needs and extra window + 1 of data because we lose the first value calculating the return"""
        return self.window + 1

    @property
    def name(self) -> str:
        return f"Volatility_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["Close"].pct_change()
        result = returns.rolling(window=self.window).std()

        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result