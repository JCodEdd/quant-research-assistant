class MarketDataError(Exception):
    """Base class for market data related exceptions."""

class InvalidTickerError(MarketDataError):
    """Raised when  ticker symbol can not be downloaded."""

class EmptyDatasetError(MarketDataError):
    """Raised when the downloaded dataset is empty."""