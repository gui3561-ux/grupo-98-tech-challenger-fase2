import pandas as pd

from preprocessing.feature_engineering import (
    encode_ids,
    temporal_split,
)
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.strategies import (
    ColdStartFilterStrategy,
    EventWeightingStrategy,
    RatingNormalizationStrategy,
)


class TestEventWeightingStrategy:
    def test_assigns_correct_weights(self, sample_events_df: pd.DataFrame) -> None:
        strategy = EventWeightingStrategy()
        result = strategy.execute(sample_events_df)
        assert "rating" in result.columns
        assert result["rating"].max() <= 3.0
        assert result["rating"].min() >= 1.0

    def test_aggregates_per_user_item(self, sample_events_df: pd.DataFrame) -> None:
        strategy = EventWeightingStrategy()
        result = strategy.execute(sample_events_df)
        duplicates = result.duplicated(subset=["visitorid", "itemid"])
        assert not duplicates.any()

    def test_takes_max_score(self) -> None:
        df = pd.DataFrame(
            {
                "visitorid": [1, 1],
                "itemid": [10, 10],
                "event": ["view", "transaction"],
                "timestamp": [1, 2],
            }
        )
        strategy = EventWeightingStrategy()
        result = strategy.execute(df)
        assert result.iloc[0]["rating"] == 3.0


class TestColdStartFilterStrategy:
    def test_filters_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "visitorid": [1, 1, 1, 2, 3, 3, 3],
                "itemid": [10, 20, 30, 10, 10, 20, 30],
                "rating": [1.0] * 7,
                "timestamp": list(range(7)),
            }
        )
        strategy = ColdStartFilterStrategy(min_interactions=3)
        result = strategy.execute(df)
        assert 2 not in result["visitorid"].values


class TestRatingNormalizationStrategy:
    def test_normalizes_to_zero_one(self) -> None:
        df = pd.DataFrame({"rating": [1.0, 2.0, 3.0]})
        strategy = RatingNormalizationStrategy()
        result = strategy.execute(df)
        assert result["rating"].min() == 0.0
        assert result["rating"].max() == 1.0

    def test_handles_uniform_ratings(self) -> None:
        df = pd.DataFrame({"rating": [2.0, 2.0, 2.0]})
        strategy = RatingNormalizationStrategy()
        result = strategy.execute(df)
        assert (result["rating"] == 1.0).all()


class TestPreprocessingPipeline:
    def test_runs_all_strategies(self, sample_events_df: pd.DataFrame) -> None:
        pipeline = PreprocessingPipeline(
            strategies=[
                EventWeightingStrategy(),
                ColdStartFilterStrategy(min_interactions=2),
                RatingNormalizationStrategy(),
            ]
        )
        result = pipeline.run(sample_events_df)
        assert len(result) > 0
        assert result["rating"].min() >= 0.0
        assert result["rating"].max() <= 1.0


class TestFeatureEngineering:
    def test_encode_ids(self, sample_interactions_df: pd.DataFrame) -> None:
        result, encoders = encode_ids(sample_interactions_df)
        assert "user_idx" in result.columns
        assert "item_idx" in result.columns
        assert result["user_idx"].min() == 0
        assert result["item_idx"].min() == 0
        assert encoders["num_users"] > 0
        assert encoders["num_items"] > 0

    def test_temporal_split_ratios(self, sample_interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(sample_interactions_df)
        total = len(sample_interactions_df)
        assert len(train) == int(total * 0.8)
        assert len(val) == int(total * 0.1)
        assert len(test) == total - int(total * 0.8) - int(total * 0.1)
