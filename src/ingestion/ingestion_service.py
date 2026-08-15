import calendar
from datetime import datetime, timezone

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
        tickers = [market_data_request.tickers]  if isinstance(market_data_request.tickers, str) else market_data_request.tickers
        start_date = pd.Timestamp(market_data_request.start_date)
        end_date = pd.Timestamp(market_data_request.end_date)

        required_months = self._get_months_in_range(start_date, end_date)

        all_ticker_data = {}

        for ticker in tickers:
            ticker_dfs = []
            missing_months = []

            for year, month in required_months:
                if self.cache.is_month_cached(market_data_request.provider, ticker, year, month):
                    ticker_dfs.append(self.cache.get_month(market_data_request.provider, ticker, year, month))
                else:
                    missing_months.append((year, month))

            if missing_months:
                # Fetch missing months and update cache
                for year, month in missing_months:
                    month_df = self._fetch_full_month(market_data_request.provider, ticker, year, month)
                    self.cache.save_month(market_data_request.provider, ticker, year, month, month_df)
                    ticker_dfs.append(month_df)

            # Combine all months for this ticker, sort, and slice to requested range
            full_ticker_df = pd.concat(ticker_dfs).sort_index()
            # Safety check to drop duplicates if any (though months shouldn't overlap)
            full_ticker_df = full_ticker_df[~full_ticker_df.index.duplicated(keep='last')]

            # Ensure index is localized
            if full_ticker_df.index.tz is None:
                full_ticker_df.index = full_ticker_df.index.tz_localize('UTC')

            target_start = start_date if start_date.tz is not None else start_date.tz_localize('UTC')
            target_end = end_date if end_date.tz is not None else end_date.tz_localize('UTC')

            all_ticker_data[ticker] = full_ticker_df.loc[target_start:target_end]

        # Combine everything into the expected format
        if len(tickers) > 1:
            result = (
                pd.concat(all_ticker_data.values(), axis=1, keys=all_ticker_data.keys(), names=['Ticker', 'Price'])
                .swaplevel(0, 1, axis=1)
            )
            return result

        return all_ticker_data[tickers[0]]

    def _get_months_in_range(
            self,
            start: pd.Timestamp,
            end: pd.Timestamp
        ) -> list[tuple[int, int]]:
        """Returns list of (year, month) tuples spanning the range"""
        months = []
        current = start.replace(day=1)
        while current <= end:
            months.append((current.year, current.month))
            current = (current + pd.DateOffset(months=1))
        return months

    def _fetch_full_month(
            self,
            provider: str,
            ticker: str,
            year: int,
            month: int
        ) -> pd.DataFrame:
        """Fetches a complete calendar month of data"""
        last_day = calendar.monthrange(year, month)[1]
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        end_date = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        request = MarketDataRequest(
            provider=provider,
            tickers=ticker,
            start_date=start_date,
            end_date=end_date
        )

        df = self.data_provider.download_data(request)

        # Normalize timezone to UTC
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        self.validator.validate_data(df)
        return df