"""Feature engineering: ID encoding and temporal train/val/test split."""

import json

import pandas as pd


def encode_ids(
    df: pd.DataFrame,
    user_col: str = "visitorid",
    item_col: str = "itemid",
) -> tuple[pd.DataFrame, dict[str, dict[int, int]]]:
    """Map user and item IDs to contiguous 0-indexed integers.

    Args:
        df: DataFrame containing the raw user/item identifier columns.
        user_col: Name of the user identifier column. Defaults to the
            RetailRocket column name for backward compatibility.
        item_col: Name of the item identifier column. Defaults to the
            RetailRocket column name for backward compatibility.

    Returns:
        Tuple of (DataFrame with user_idx/item_idx added, encoders dict).
    """
    df = df.copy()
    user_ids = sorted(df[user_col].unique())
    item_ids = sorted(df[item_col].unique())
    user_encoder = {uid: idx for idx, uid in enumerate(user_ids)}
    item_encoder = {iid: idx for idx, iid in enumerate(item_ids)}
    df["user_idx"] = df[user_col].map(user_encoder)
    df["item_idx"] = df[item_col].map(item_encoder)
    encoders = {
        "user_encoder": {str(k): v for k, v in user_encoder.items()},
        "item_encoder": {str(k): v for k, v in item_encoder.items()},
        "num_users": len(user_ids),
        "num_items": len(item_ids),
    }
    return df, encoders


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by timestamp into train/val/test sets.

    Args:
        df: DataFrame with a 'timestamp' column.
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction of rows for validation.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def save_encoders(encoders: dict, path: str) -> None:
    """Save encoder mappings to a JSON file.

    Args:
        encoders: Dict produced by encode_ids().
        path: Destination file path.
    """
    with open(path, "w") as f:
        json.dump(encoders, f, indent=2)


def load_encoders(path: str) -> dict:
    """Load encoder mappings from a JSON file.

    Args:
        path: Source file path.

    Returns:
        Dict with user_encoder, item_encoder, num_users, num_items.
    """
    with open(path) as f:
        return json.load(f)
