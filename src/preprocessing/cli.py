from pathlib import Path

import pandas as pd

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.strategies import (
    ColdStartFilterStrategy,
    RatingNormalizationStrategy,
)
from utils import Settings, get_logger, set_global_seed

logger = get_logger(__name__)


def load_movielens(raw_path: Path) -> pd.DataFrame:
    """Load the MovieLens 100K ratings file.

    Returns a DataFrame with visitorid, itemid, rating, timestamp columns,
    matching the column names expected by feature_engineering.py.
    """
    path = raw_path / "ml-100k" / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df.rename(columns={"user_id": "visitorid", "item_id": "itemid"})


def main() -> None:
    """Preprocess raw MovieLens ratings into cleaned interactions."""
    settings = Settings()
    set_global_seed(settings.random_seed)

    output_path = settings.data_processed_path / "interactions.parquet"

    logger.info("Loading MovieLens ratings from %s", settings.data_raw_path)
    df = load_movielens(settings.data_raw_path)
    logger.info("Loaded %d ratings", len(df))

    pipeline = PreprocessingPipeline(
        strategies=[
            ColdStartFilterStrategy(settings.min_interactions),
            RatingNormalizationStrategy(),
        ]
    )

    df = pipeline.run(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d interactions to %s (users=%d, items=%d)",
        len(df),
        output_path,
        df["visitorid"].nunique(),
        df["itemid"].nunique(),
    )


if __name__ == "__main__":
    main()
