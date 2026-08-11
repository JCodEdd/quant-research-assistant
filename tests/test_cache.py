from datetime import datetime, timezone

import pandas as pd
import pytest

from ingestion.cache import Cache
from models.market_data_request import MarketDataRequest


@pytest.fixture
def sample_request():
    return MarketDataRequest(
        provider="yahoo",
        ticker="BTC-USD",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 1, 10, tzinfo=timezone.utc)
    )

@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=3)
    return pd.DataFrame({
        "Open": [10, 11, 12],
        "Close": [11, 12, 13]
    }, index=dates)

def test_cache_save_load_exists(tmp_path, sample_request, sample_df):
    cache = Cache(tmp_path)

    # Initially, the cache should not have the data
    assert cache.exists_in_cache(sample_request) is False

    # Save the data to the cache
    cache.save_to_cache(sample_df, sample_request)

    # Now, the cache should have the data
    assert cache.exists_in_cache(sample_request) is True

    # Now, the cache should have the data
    loaded_df = cache.load_from_cache(sample_request)
    pd.testing.assert_frame_equal(sample_df, loaded_df, check_freq=False)