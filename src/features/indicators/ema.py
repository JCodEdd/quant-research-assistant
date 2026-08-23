import pandas as pd

from features.indicator import Indicator


class EMA(Indicator):
    """Exponential Moving Average (EMA)"""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Uses a 3x window multiplier to dilute the error (initialization buffer)"""
        return self.window * 3

    @property
    def name(self) -> str:
        return f"EMA_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df["Close"].ewm(span=self.window, adjust=False).mean()
        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([self.name], result.columns)
        return result