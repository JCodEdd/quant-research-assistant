from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from features.feature_pipeline import FeaturePipeline
from features.indicators.rsi import RSI
from features.indicators.sma import SMA
from features.indicators.volatility import RollingVolatility
from ingestion.cache import Cache
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from ingestion.providers.yahoo_provider import YahooFinanceProvider
from models.market_data_request import MarketDataRequest

provider = YahooFinanceProvider()

cache = Cache(
    Path("data/cache")
)

validator = DataValidator()

service = IngestionService(
    data_provider=provider,
    cache=cache,
    validator=validator
)

# 1. Fetch multi-ticker data
request = MarketDataRequest(
    provider="yahoo",
    tickers= ["AAPL", "MSFT"],
    start_date=datetime(2023,1,1,tzinfo=timezone.utc),
    end_date=datetime(2023,1,31,tzinfo=timezone.utc)
)

df = service.get_data(request)
print("--- Raw Data (Head) ---")
print(df.head())

# 2. Define pipeline
pipeline = FeaturePipeline(indicators=[
    SMA(window=20),
    RSI(window=14),
    RollingVolatility(window=20)
])

# 3. Apply pipeline
enriched_df = pipeline.run(df)

print("\n--- Enriched Data (Indicators) ---")
# Show a snippet of the columns and the values
if isinstance(enriched_df.columns, pd.MultiIndex):
    print(enriched_df.columns.levels[0].tolist())
print(enriched_df.tail())
