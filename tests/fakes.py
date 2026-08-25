from datetime import timezone
from pathlib import Path

import pandas as pd

from features.indicator import Indicator
from ingestion.cache import Cache
from interfaces.data_provider import DataProvider
from models.market_data_request import MarketDataRequest


class InMemoryCache(Cache):
    """A lightweight in memory fake for the Cache."""

    def __init__(self):
        self._store = {}

    def is_month_cached(self, provider: str, ticker: str, year: int, month: int) -> bool:
        return (provider, ticker, year, month) in self._store

    def get_month(self, provider: str, ticker: str, year: int, month: int) -> pd.DataFrame | None:
        return self._store.get((provider, ticker, year, month))

    def save_month(
        self,
        provider: str,
        ticker: str,
        year: int,
        month: int,
        df: pd.DataFrame
    ) -> None:
        self._store[(provider, ticker, year, month)] = df.copy()


class FakeProvider(DataProvider):
    """A fake data provider that creates synthetic market data and tracks downloads."""

    def __init__(self):
        self.download_count = 0

    def download_data(self, market_data_request: MarketDataRequest) -> pd.DataFrame:
        self.download_count += 1

        tickers = [market_data_request.tickers] if isinstance(market_data_request.tickers, str) else market_data_request.tickers
        dates = pd.date_range(
            market_data_request.start_date,
            market_data_request.end_date,
            tz=timezone.utc
        )

        if len(tickers) == 1:
            data = {
                'Open': [100.0 + i for i in range(len(dates))],
                'High': [105.0 + i for i in range(len(dates))],
                'Low': [95.0 + i for i in range(len(dates))],
                'Close': [102.0 + i for i in range(len(dates))],
                'Volume': [1000 + i * 10 for i in range(len(dates))]
            }
            return pd.DataFrame(data, index=dates)
        else:
            metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
            columns = pd.MultiIndex.from_product([metrics, tickers])
            data = [[100] * len(columns) for _ in range(len(dates))]
            return pd.DataFrame(data, index=dates, columns=columns)


class EmptyFakeProvider(DataProvider):
    """Fake provider simulating an asset with no data found."""

    def download_data(self, market_data_request: MarketDataRequest) -> pd.DataFrame:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


class DummyIndicator(Indicator):
    """Dummy indicator to isolate the pipeline from specific indicators logic"""

    def __init__(self, window: int) -> None:
        self.window = window

    @property
    def required_lookback(self) -> int:
        return self.window

    @property
    def name(self) -> str:
        return f"DummyIndicator_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df['Close']
        if isinstance(result, pd.Series):
            return result.to_frame(self.name)
        result.columns = pd.MultiIndex.from_product([[self.name], result.columns])
        return result
