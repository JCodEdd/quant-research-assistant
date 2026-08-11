import pandas as pd

from exceptions.data_exceptions import (
    EmptyDatasetError,
    MarketDataError,
)


class DataValidator:
    """
    Validates the integrity of market data.
    """
    @staticmethod
    def validate_data(
        data: pd.DataFrame
    ) -> None:
        """
        Validate the integrity of the provided market data.

        Raises:
            MarketDataError: If the data is empty or contains NaN values.
        """
        required_columns = {
            "Open", 
            "High", 
            "Low", 
            "Close", 
            "Volume"
        }

        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise MarketDataError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )

        if data.empty:
            raise EmptyDatasetError(
                "Data is empty."
            )

        if data.index.has_duplicates:
            raise MarketDataError(
                "Data contains duplicate timestamps."
            )

        if data.index.is_monotonic_increasing is False:
            raise MarketDataError(
                "Data timestamps are not in chronological order."
            )
        if data.isnull().any().any():
            raise MarketDataError(
                "Data contains NaN values."
            )
       