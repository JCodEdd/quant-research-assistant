import pandas as pd

from features.indicator import Indicator


class RSI(Indicator):
    """Relative Strength Index (RSI)"""

    def __init__(self, window: int = 14):
        self.window = window

    @property
    def name(self) -> str:
        return f"RSI_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing via ewm
        avg_gain = gain.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi