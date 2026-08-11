from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketDataRequest:
    """
    Data class representing a market data request.
    """
    provider: str
    ticker: str
    start_date: datetime
    end_date: datetime