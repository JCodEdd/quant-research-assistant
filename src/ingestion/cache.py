from pathlib import Path

import pandas as pd

from models.market_data_request import MarketDataRequest


class Cache:

    def __init__(
        self, 
        cache_directory: Path
    ):
        self.cache_directory = cache_directory

    def exists_in_cache(
        self, 
        request: MarketDataRequest
    ) -> bool:
        return self._build_cache_path(request).exists()

    def load_from_cache(
        self, 
        request: MarketDataRequest
    ) -> pd.DataFrame:
        path = self._build_cache_path(request)
        return pd.read_parquet(path)

    def save_to_cache(
        self, 
        dataFrame: pd.DataFrame, 
        request: MarketDataRequest
    ):
        path = self._build_cache_path(request)
        dataFrame.to_parquet(path)

    def _build_cache_path(
        self, 
        request: MarketDataRequest
    ) -> Path:
        filename =( 
            f"{request.provider}_"
            f"{request.ticker}_"
            f"{request.start_date.strftime('%Y-%m-%d')}_"
            f"{request.end_date.strftime('%Y-%m-%d')}.parquet"
        )
        return self.cache_directory / filename
