import pandas as pd

from features.indicator import Indicator


class RSI(Indicator):
    """Relative Strength Index (RSI)"""

    def __init__(self, window: int = 14):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Uses a 5x window multiplier to dilute the error (initialization buffer)"""
        return self.window * 5

    @property
    def name(self) -> str:
        return f"RSI_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing via ewm
        avg_gain = gain.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if isinstance(rsi, pd.Series):
                return rsi.to_frame(self.name)

        rsi.columns = pd.MultiIndex.from_product([[self.name], rsi.columns])
        return rsi