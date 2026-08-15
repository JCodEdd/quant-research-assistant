from pathlib import Path
from typing import Optional

import pandas as pd

from models.market_data_request import MarketDataRequest


class Cache:

    def __init__(
        self, 
        cache_directory: Path
    ):
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def is_month_cached(self, provider: str, ticker: str, year: int, month: int) -> bool:
        return self._get_path(provider, ticker, year, month).exists()

    def get_month(
            self,
            provider: str,
            ticker: str,
            year: int,
            month: int
    ) -> Optional[pd.DataFrame]:
        path = self._get_path(provider, ticker, year, month)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save_month(
            self,
            provider: str,
            ticker: str,
            year: int,
            month: int,
            df: pd.DataFrame
    ):
        path = self._get_path(provider, ticker, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def _get_path(self, provider: str, ticker: str, year: int, month: int) -> Path:
        filename = f"{year:04d}-{month:02d}.parquet"
        return self.cache_directory / provider / ticker / filename