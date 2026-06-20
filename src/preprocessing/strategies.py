from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    """Abstract base for preprocessing steps (Strategy pattern)."""

    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the DataFrame."""


class EventWeightingStrategy(PreprocessingStrategy):
    """Assign numeric weights to event types and aggregate per user-item."""

    EVENT_WEIGHTS: dict[str, float] = {
        "view": 1.0,
        "addtocart": 2.0,
        "transaction": 3.0,
    }

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map events to weights and keep max score per user-item pair."""
        df = df.copy()
        df["rating"] = df["event"].map(self.EVENT_WEIGHTS)
        df = df.dropna(subset=["rating"])
        return df.groupby(["visitorid", "itemid"], as_index=False).agg(
            {"rating": "max", "timestamp": "max"}
        )


class ColdStartFilterStrategy(PreprocessingStrategy):
    """Remove users and items with fewer than min_interactions."""

    def __init__(self, min_interactions: int) -> None:
        self.min_interactions = min_interactions

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iteratively filter until all users/items meet threshold."""
        df = df.copy()
        while True:
            user_counts = df["visitorid"].value_counts()
            item_counts = df["itemid"].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            valid_items = item_counts[item_counts >= self.min_interactions].index
            filtered = df[
                df["visitorid"].isin(valid_users) & df["itemid"].isin(valid_items)
            ]
            if len(filtered) == len(df):
                break
            df = filtered
        return df


class RatingNormalizationStrategy(PreprocessingStrategy):
    """Normalize ratings to [0, 1] using min-max scaling."""

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization to the rating column."""
        df = df.copy()
        min_val = df["rating"].min()
        max_val = df["rating"].max()
        if max_val > min_val:
            df["rating"] = (df["rating"] - min_val) / (max_val - min_val)
        else:
            df["rating"] = 1.0
        return df
