from datetime import datetime, timezone
from pathlib import Path

from features.indicators.rsi import RSI
from features.indicators.sma import SMA
from features.indicators.volatility import RollingVolatility
from ingestion.cache import Cache
from ingestion.data_validator import DataValidator
from ingestion.ingestion_service import IngestionService
from ingestion.providers.yahoo_provider import YahooFinanceProvider
from research.research_orchestrator import ResearchOrchestrator

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

orchestrator = ResearchOrchestrator(service)

# Define request parameters
tickers = ["AAPL", "MSFT"]
start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2023, 6, 1, tzinfo=timezone.utc)
indicators = [
    SMA(window=20),
    RSI(window=14),
    RollingVolatility(window=20)
]

# Run orchestration
print(f"--- Fetching enriched data from {tickers} ---")
enriched_df = orchestrator.get_enriched_data(
    provider="yahoo",
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    indicators=indicators
)

print("\n--- Enriched Data (Head)")
print(enriched_df.head())

print("\n--- NaN Check on First Requested Date ---")
print(enriched_df.iloc[0].isna().sum(), "NaN values found on start date (Warmed-up!)")
