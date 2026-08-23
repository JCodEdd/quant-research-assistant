import pandas as pd

from features.indicator import Indicator


class RollingLow(Indicator):
    """Rolling Minimum Low over a window"""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Only needs the an extra window of previous"""
        return self.window

    @property
    def name(self) -> str:
        return f"RollingLow_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df["Low"].rolling(window=self.window).min()

        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.column = pd.MultiIndex.from_product([[self.name], result.columns])
        return result