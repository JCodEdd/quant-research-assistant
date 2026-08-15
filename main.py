from datetime import datetime, timezone
from pathlib import Path

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

request = MarketDataRequest(
    provider="yahoo",
    tickers= ["AAPL", "MSFT"],
    start_date=datetime(2023,1,1,tzinfo=timezone.utc),
    end_date=datetime(2023,1,31,tzinfo=timezone.utc)
)

df = service.get_data(request)
print(df.head())
print(f"Columns: {df.columns.to_list()}")