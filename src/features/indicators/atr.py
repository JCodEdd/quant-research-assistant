import numpy as np
import pandas as pd

from features.indicator import Indicator


class ATR(Indicator):
    """Average True Range (ATR)"""

    def __init__(self, window: int = 14):
        self.window = window

    @property
    def required_lookback(self) -> int:
        """Uses a 5x window multiplier to dilute the error (initialization buffer)"""
        return self.window * 5

    @property
    def name(self) -> str:
        return f"ATR_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # Element-wise maximum, skipping NaNs
        tr = tr1.combine(tr2, np.fmax).combine(tr3, np.fmax)

        result = tr.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()

        if isinstance(result, pd.Series):
            return result.to_frame(self.name)

        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result