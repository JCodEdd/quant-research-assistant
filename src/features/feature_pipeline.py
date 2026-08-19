import pandas as pd

from features.indicator import Indicator


class FeaturePipeline:
    """Pipeline to apply a list of idicators and enrich a market data DataFrame"""

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
        enriched_df = df.copy()
        for indicator in self.indicators:
            enriched_df[indicator.name] = indicator.compute(df)
        return enriched_df