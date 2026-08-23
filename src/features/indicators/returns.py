import pandas as pd

from features.indicator import Indicator


class Returns(Indicator):
    """Percentage returns over a rolling window."""

    def __init__(self, window: int = 1):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Only needs an extra window of previous data"""
        return self.window

    @property
    def name(self) -> str:
        return f"Returns_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df["Close"].pct_change(periods=self.window)

        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result