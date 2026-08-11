from abc import ABC, abstractmethod

import pandas as pd

from models.market_data_request import MarketDataRequest


class DataProvider(ABC):
    """
    Base interface for market data providers.
    """
    @abstractmethod
    def download_data(
        self,
        market_data_request: MarketDataRequest,
    ) -> pd.DataFrame:
        """
        Download historical market data for a given ticker and date range.
        """
    