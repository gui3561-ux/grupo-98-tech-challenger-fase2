"""Compares MLP against Popularity and SVD baselines on the test set.

Trains the baselines (which have no gradient-based training loop),
evaluates all three models with the same >= 4 metrics, and logs each
as a separate MLflow run for side-by-side comparison in the UI.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import ModelEvaluator  # noqa: E402
from models.factory import ModelFactory  # noqa: E402
from preprocessing.feature_engineering import load_encoders  # noqa: E402
from training.dataset import InteractionDataset  # noqa: E402
from training.tracker import ExperimentTracker  # noqa: E402
from utils import Settings, get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)


def build_user_items_map(df: pd.DataFrame) -> dict[int, set[int]]:
    """Build a mapping of user_idx to the set of relevant item_idx.

    Args:
        df: DataFrame with user_idx and item_idx columns.

    Returns:
        Dict mapping each user index to the set of items they interacted with.
    """
    user_items: dict[int, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        user_items[int(row["user_idx"])].add(int(row["item_idx"]))
    return dict(user_items)


def evaluate_and_log(
    name: str,
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_user_items: dict[int, set[int]],
    num_items: int,
    tracker: ExperimentTracker,
) -> dict[str, float]:
    """Evaluate a model and log its metrics as a separate MLflow run.

    Args:
        name: Model display name, used as the MLflow run name.
        model: Trained model implementing forward(user_ids, item_ids).
        test_loader: DataLoader over the test set.
        test_user_items: Mapping of user_idx to relevant item_idx set.
        num_items: Total number of items in the catalogue.
        tracker: Shared ExperimentTracker instance.

    Returns:
        Dict of metric name to value for this model.
    """
    evaluator = ModelEvaluator(model, top_k=10)
    metrics = evaluator.evaluate(test_loader, test_user_items, num_items)

    tracker.start_run(run_name=f"compare-{name}")
    tracker.log_params({"model": name})
    tracker.log_metrics(metrics)
    tracker.end_run()

    logger.info("%s metrics: %s", name, metrics)
    return metrics


def main() -> None:
    """Train baselines, evaluate all models, and log comparison to MLflow."""
    settings = Settings()
    set_global_seed(settings.random_seed)

    features_path = settings.data_features_path
    encoders = load_encoders(str(features_path / "encoders.json"))
    train_df = pd.read_parquet(features_path / "train.parquet")
    test_df = pd.read_parquet(features_path / "test.parquet")

    num_users = encoders["num_users"]
    num_items = encoders["num_items"]

    test_dataset = InteractionDataset(
        test_df["user_idx"].values,
        test_df["item_idx"].values,
        test_df["rating"].values,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=settings.batch_size, shuffle=False
    )
    test_user_items = build_user_items_map(test_df)

    tracker = ExperimentTracker(settings)
    results: dict[str, dict[str, float]] = {}

    # --- MLP (already trained) ---
    mlp = ModelFactory.create(
        "mlp",
        num_users=num_users,
        num_items=num_items,
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims_list,
    )
    mlp_path = settings.models_path / "model.pt"
    mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
    logger.info("Loaded trained MLP from %s", mlp_path)
    results["mlp"] = evaluate_and_log(
        "mlp", mlp, test_loader, test_user_items, num_items, tracker
    )

    # --- Popularity baseline ---
    popularity = ModelFactory.create(
        "popularity", num_users=num_users, num_items=num_items
    )
    popularity.fit(train_df["item_idx"].values)
    results["popularity"] = evaluate_and_log(
        "popularity", popularity, test_loader, test_user_items, num_items, tracker
    )

    # --- SVD baseline ---
    svd = ModelFactory.create(
        "svd", num_users=num_users, num_items=num_items, n_components=50
    )
    svd.fit(
        train_df["user_idx"].values,
        train_df["item_idx"].values,
        train_df["rating"].values,
    )
    results["svd"] = evaluate_and_log(
        "svd", svd, test_loader, test_user_items, num_items, tracker
    )

    # --- Save comparison report ---
    report_path = settings.models_path / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved comparison report to %s", report_path)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 60)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
