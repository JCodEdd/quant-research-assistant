import pandas as pd
import yfinance as yf

from exceptions.data_exceptions import   EmptyDatasetError

from interfaces.data_provider import DataProvider
from models.market_data_request import MarketDataRequest
from utils.logger import logger


class YahooFinanceProvider(DataProvider):
    """
    Data provider implementation for Yahoo Finance.
    """
    def download_data(
        self,
        market_data_request: MarketDataRequest
    ) -> pd.DataFrame:
        """
        Load historical market data from Yahoo Finance for a given ticker and date range.
        """

        logger.info(f"Downloading data for {market_data_request.tickers} from {market_data_request.start_date} to {market_data_request.end_date}")

        data = yf.download(
            tickers=market_data_request.tickers,
            start=market_data_request.start_date,
            end=market_data_request.end_date,
            progress=False
        )

        if data is None or data.empty:
            raise EmptyDatasetError(
                f"No data found for ticker {market_data_request.tickers}"
            )

        # Convert multi-index to single index when single ticker
        if isinstance(market_data_request.tickers, str):
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
        else:
            pass

        logger.info(
            f"Downloaded {len(data)} rows of data for {market_data_request.tickers}"
        )    

        return data