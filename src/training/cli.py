import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.factory import ModelFactory
from preprocessing.feature_engineering import load_encoders
from training.dataset import InteractionDataset
from training.tracker import ExperimentTracker
from training.trainer import Trainer
from utils import Settings, get_logger, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Train the recommendation model."""

    logger.info("-----Start Training---")
    settings = Settings()
    set_global_seed(settings.random_seed)

    features_path = settings.data_features_path
    encoders = load_encoders(str(features_path / "encoders.json"))

    train_df = pd.read_parquet(features_path / "train.parquet")
    val_df = pd.read_parquet(features_path / "val.parquet")

    train_dataset = InteractionDataset(
        train_df["user_idx"].values,
        train_df["item_idx"].values,
        train_df["rating"].values,
    )
    val_dataset = InteractionDataset(
        val_df["user_idx"].values,
        val_df["item_idx"].values,
        val_df["rating"].values,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=settings.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False)

    model = ModelFactory.create(
        "mlp",
        num_users=encoders["num_users"],
        num_items=encoders["num_items"],
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims_list,
    )

    tracker = ExperimentTracker(settings)
    tracker.start_run(run_name="mlp-training")
    tracker.log_params(
        {
            "model": "mlp",
            "num_users": encoders["num_users"],
            "num_items": encoders["num_items"],
            "embedding_dim": settings.embedding_dim,
            "hidden_dims": settings.hidden_dims,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "num_epochs": settings.num_epochs,
            "early_stopping_patience": settings.early_stopping_patience,
            "random_seed": settings.random_seed,
        }
    )

    trainer = Trainer(model, settings)
    history = trainer.fit(train_loader, val_loader)

    for epoch, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
        tracker.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

    models_path = settings.models_path
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to %s", model_path)

    metrics = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_trained": len(history["train_loss"]),
    }
    metrics_path = models_path / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    tracker.log_model(model, name="mlp-recommender")
    tracker.end_run()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
