import pandas as pd

from preprocessing.strategies import PreprocessingStrategy
from utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Executes a sequence of preprocessing strategies."""

    def __init__(self, strategies: list[PreprocessingStrategy]) -> None:
        self.strategies = strategies

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all strategies in order, logging progress."""
        for strategy in self.strategies:
            name = type(strategy).__name__
            logger.info("Running %s...", name)
            df = strategy.execute(df)
            logger.info("After %s: %d rows", name, len(df))
        return df
