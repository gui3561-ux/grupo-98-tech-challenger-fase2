import pandas as pd

from preprocessing.feature_engineering import (
    encode_ids,
    save_encoders,
    temporal_split,
)
from utils import Settings, get_logger, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Engineer features: encode IDs, split train/val/test."""
    settings = Settings()
    set_global_seed(settings.random_seed)

    input_path = settings.data_processed_path / "interactions.parquet"
    features_path = settings.data_features_path

    logger.info("Loading processed interactions from %s", input_path)
    df = pd.read_parquet(input_path)

    df, encoders = encode_ids(df)
    logger.info(
        "Encoded %d users, %d items",
        encoders["num_users"],
        encoders["num_items"],
    )

    train_df, val_df, test_df = temporal_split(df)
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    features_path.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(features_path / "train.parquet", index=False)
    val_df.to_parquet(features_path / "val.parquet", index=False)
    test_df.to_parquet(features_path / "test.parquet", index=False)
    save_encoders(encoders, str(features_path / "encoders.json"))
    logger.info("Saved features to %s", features_path)


if __name__ == "__main__":
    main()
