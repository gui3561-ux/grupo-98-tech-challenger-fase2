import pandas as pd

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.strategies import (
    ColdStartFilterStrategy,
    EventWeightingStrategy,
    RatingNormalizationStrategy,
)
from utils import Settings, get_logger, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Preprocess raw events into cleaned interactions."""
    settings = Settings()
    set_global_seed(settings.random_seed)

    raw_path = settings.data_raw_path / "events.csv"
    output_path = settings.data_processed_path / "interactions.parquet"

    logger.info("Loading raw events from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d events", len(df))

    pipeline = PreprocessingPipeline(
        strategies=[
            EventWeightingStrategy(),
            ColdStartFilterStrategy(settings.min_interactions),
            RatingNormalizationStrategy(),
        ]
    )

    df = pipeline.run(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d interactions to %s", len(df), output_path)


if __name__ == "__main__":
    main()
