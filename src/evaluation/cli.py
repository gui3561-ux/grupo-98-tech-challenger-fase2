import json
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation.evaluator import ModelEvaluator
from models.factory import ModelFactory
from preprocessing.feature_engineering import load_encoders
from training.dataset import InteractionDataset
from training.tracker import ExperimentTracker
from utils import Settings, get_logger, set_global_seed

logger = get_logger(__name__)


def build_user_items_map(df: pd.DataFrame) -> dict[int, set[int]]:
    """Build mapping of user_idx to set of relevant item_idx."""
    user_items: dict[int, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        user_items[int(row["user_idx"])].add(int(row["item_idx"]))
    return dict(user_items)


def main() -> None:
    """Evaluate the trained model on test data."""
    settings = Settings()
    set_global_seed(settings.random_seed)

    features_path = settings.data_features_path
    encoders = load_encoders(str(features_path / "encoders.json"))

    test_df = pd.read_parquet(features_path / "test.parquet")
    test_dataset = InteractionDataset(
        test_df["user_idx"].values,
        test_df["item_idx"].values,
        test_df["rating"].values,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=settings.batch_size, shuffle=False
    )

    model = ModelFactory.create(
        "mlp",
        num_users=encoders["num_users"],
        num_items=encoders["num_items"],
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims_list,
    )

    model_path = settings.models_path / "model.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Loaded model from %s", model_path)

    test_user_items = build_user_items_map(test_df)
    evaluator = ModelEvaluator(model, top_k=10)
    metrics = evaluator.evaluate(test_loader, test_user_items, encoders["num_items"])

    metrics_path = settings.models_path / "eval_metrics.json"
    print('-->', metrics_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)

    tracker = ExperimentTracker(settings)
    tracker.start_run(run_name="mlp-evaluation")
    print(' Metrics --> ',metrics)
    tracker.log_metrics(metrics)
    tracker.log_artifact(str(metrics_path))
    tracker.end_run()


if __name__ == "__main__":
    main()
