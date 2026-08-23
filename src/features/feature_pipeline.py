import pandas as pd

from features.indicator import Indicator


class FeaturePipeline:
    """Pipeline to apply a list of indicators and enrich a market data DataFrame"""

    def __init__(self, indicators: list[Indicator]) -> None:
        self.indicators = indicators

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the imput DataFrame with calculated indicators.

        Args:
            df (pd.DataFrame): Input market data DataFrame.

        Returns:
            pd.DataFrame: DataFrame enriched with indicator columns.
        """
        results = [df]
        for indicator in self.indicators:
            results.append(indicator.compute(df))
        return pd.concat(results, axis=1)