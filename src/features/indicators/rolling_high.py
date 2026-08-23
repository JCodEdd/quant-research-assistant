import pandas as pd

from features.indicator import Indicator


class RollingHigh(Indicator):
    """Rolling Maximum High over a window."""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Only needs an extra window of data"""
        return self.window

    @property
    def name(self) -> str:
        return f"RollingHigh_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df["High"].rolling(window=self.window).max()

        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result