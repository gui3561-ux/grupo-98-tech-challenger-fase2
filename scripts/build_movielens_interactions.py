"""Builds data/processed/interactions.parquet from MovieLens 100k.

This produces the same intermediate artifact that the RetailRocket
preprocessing/cli.py would produce, so that the existing
preprocessing/feature_cli.py can run unmodified afterwards.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import Settings, get_logger

logger = get_logger(__name__)


def load_movielens(raw_path: Path) -> pd.DataFrame:
    """Load the MovieLens 100K ratings file.

    Args:
        raw_path: Path to the data/raw directory.

    Returns:
        DataFrame with visitorid, itemid, rating, timestamp columns,
        matching the column names expected by feature_cli.py.
    """
    path = raw_path / "ml-100k" / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Rename to match the column names feature_engineering.py expects
    df = df.rename(columns={"user_id": "visitorid", "item_id": "itemid"})

    # Normalise ratings to [0, 1] — MLP head uses Sigmoid output
    min_r, max_r = df["rating"].min(), df["rating"].max()
    df["rating"] = (df["rating"] - min_r) / (max_r - min_r)

    logger.info("Loaded %d ratings from %s", len(df), path)
    return df


def main() -> None:
    """Build interactions.parquet for the MovieLens dataset."""
    settings = Settings()
    df = load_movielens(Path(settings.data_raw_path))

    processed_path = Path(settings.data_processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    output_path = processed_path / "interactions.parquet"

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
