import pandas as pd

from features.indicator import Indicator


class ATR(Indicator):
    """Average True Range (ATR)"""

    def __init__(self, window: int = 14):
        self.window = window

    @property
    def name(self) -> str:
        return f"ATR_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()