import pandas as pd
import pytest

from exceptions.data_exceptions import EmptyDatasetError, MarketDataError
from ingestion.data_validator import DataValidator


@pytest.fixture
def valid_df():
    dates = pd.date_range(start="2023-01-01", periods=5)
    return pd.DataFrame({
        "Open": [10, 11, 12, 13, 14],
        "High": [11, 12, 13, 14, 15],
        "Low": [9, 10, 11, 12, 13],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    }, index=dates)


# ============================================================================
# Standard Scenario
# ============================================================================

def test_validator_valid_data(valid_df):
    """Validates that a complete OHLCV market DataFrame passes validation with no errors."""
    # Should not raise any exceptions
    DataValidator.validate_data(valid_df)

# ============================================================================
# Failure Scenarios
# ============================================================================

def test_validator_missing_columns(valid_df):
    """Validates that missing any required price or volume column raises a MarketDataError."""
    # Arrange
    df_missing = valid_df.drop(columns=["Volume"])

    # Act & Assert
    with pytest.raises(MarketDataError, match="Missing required columns: Volume"):
        DataValidator.validate_data(df_missing)


def test_validator_contains_nan_values(valid_df):
    """Validates that a DataFrame containing NaN/null values raises a MarketDataError."""
    # Arrange
    valid_df.loc[valid_df.index[2], "Open"] = None

    # Act & Assert
    with pytest.raises(MarketDataError, match="Data contains NaN values."):
        DataValidator.validate_data(valid_df)

def test_validator_duplicate_timestamps(valid_df):
    """Validates that a DataFrame containing duplicate timestamps raises a MarketDataError."""
    # Introduce duplicate timestamps
    df_duplicate = pd.concat([valid_df.iloc[[0]], valid_df])

    # Act & Assert
    with pytest.raises(MarketDataError, match="Data contains duplicate timestamps."):
        DataValidator.validate_data(df_duplicate)

def test_validator_unsorted_timestamps(valid_df):
    """Validates that a DataFrame containing unsorted timestamps raises a MarketDataError."""
    # Shuffle the DataFrame to make timestamps unsorted
    df_unsorted = valid_df.sample(frac=1)

    # Act & Assert
    with pytest.raises(MarketDataError, match="Data timestamps are not in chronological order."):
        DataValidator.validate_data(df_unsorted)

# ============================================================================
# Edge Cases
# ============================================================================

def test_validator_multi_index_columns():
    """Validates that a MultiIndex panel DataFrame (multi-ticker format) passes validation."""
    # Arrange
    dates = pd.date_range(start="2023-01-01", periods=2)
    # Create a multiIndex dataframe (Metric, Ticker)
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]])
    data = [[1.0]*10, [2.0]*10]
    df_multi_index = pd.DataFrame(data, index=dates, columns=columns)

    # Act & Assert
    # Should not raise any exceptions
    DataValidator.validate_data(df_multi_index)

def test_validator_empty_dataset():
    """Validates that an empty DataFrame raises an EmptyDatasetError."""
    # Arrange
    df_empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Act & Assert
    with pytest.raises(EmptyDatasetError, match="Data is empty."):
        DataValidator.validate_data(df_empty)
