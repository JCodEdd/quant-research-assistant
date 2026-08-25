
from datetime import datetime, timedelta

import pandas as pd

from features.feature_pipeline import FeaturePipeline
from features.indicator import Indicator
from ingestion.ingestion_service import IngestionService
from models.market_data_request import MarketDataRequest


class ResearchOrchestrator:
    """Orchestrates data ingestion with lookback expansion and feature pipeline execution."""

    def __init__(self, ingestion_service: IngestionService) -> None:
        self.ingestion_service = ingestion_service

    def get_enriched_data(
            self,
            provider: str,
            tickers: list[str] | str,
            start_date: datetime,
            end_date: datetime,
            indicators: list[Indicator]
    ) -> pd.DataFrame:
        """
        Fetches market data with extended lookback, runs the feature pipeline,
        and slices the result back to the requested [start_date, end_date] range
        """
        pipeline = FeaturePipeline(indicators)
        max_lookback = pipeline.get_max_lookback()

        # Convert bar lookback to calendar days buffer (approx 1.5x factor + safety margin)
        # We assume daily data for now.
        calendar_buffer_days = int(max_lookback * 1.5) + 15
        extended_start_date = start_date - timedelta(days=calendar_buffer_days)

        request = MarketDataRequest(
            provider=provider,
            tickers=tickers,
            start_date=extended_start_date,
            end_date=end_date
        )

        extended_df = self.ingestion_service.get_data(request)

        # Run pipeline on extended dataset
        enriched_extended_df = pipeline.run(extended_df)

        # Slice back to original requested date range
        # Assume input DataFrame index is a DatetimeIndex
        result_df = enriched_extended_df.loc[start_date:end_date]

        return result_df