import pandas as pd

from ingestion.cache import Cache
from ingestion.data_validator import DataValidator
from interfaces.data_provider import DataProvider
from models.market_data_request import MarketDataRequest


class IngestionService:

    def __init__(
            self, 
            data_provider: DataProvider,
            cache: Cache,
            validator: DataValidator
        ):
            self.data_provider = data_provider
            self.cache = cache
            self.validator = validator


    def get_data(
        self, 
        market_data_request: MarketDataRequest
    ) -> pd.DataFrame:
        """
        Get historical market data for a given ticker and date range.
        """
        if self.cache.exists_in_cache(market_data_request):
            dataframe = self.cache.load_from_cache(market_data_request)
        else:
            dataframe = self.data_provider.download_data(market_data_request)
            self.validator.validate_data(dataframe)
            self.cache.save_to_cache(dataframe, market_data_request)

        return dataframe.copy()