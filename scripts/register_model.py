"""Registers the trained MLP in the MLflow Model Registry.

Registers a new model version, then promotes it through the
Staging -> Production lifecycle using MLflow's alias system
(the modern replacement for the deprecated stage-based API).
"""

import sys
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.factory import ModelFactory  # noqa: E402
from preprocessing.feature_engineering import load_encoders  # noqa: E402
from utils import Settings, get_logger  # noqa: E402

logger = get_logger(__name__)

REGISTERED_MODEL_NAME = "ecommerce-mlp-recommender"


def load_trained_model(settings: Settings) -> torch.nn.Module:
    """Load the trained MLP from disk.

    Args:
        settings: Application settings.

    Returns:
        MLPRecommender with weights loaded from models/model.pt.
    """
    features_path = settings.data_features_path
    encoders = load_encoders(str(features_path / "encoders.json"))

    model = ModelFactory.create(
        "mlp",
        num_users=encoders["num_users"],
        num_items=encoders["num_items"],
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims_list,
    )
    model_path = settings.models_path / "model.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Loaded trained model from %s", model_path)
    return model


def log_and_register(model: torch.nn.Module, settings: Settings) -> str:
    """Log the model to MLflow and register it in the Model Registry.

    Args:
        model: Trained PyTorch model to register.
        settings: Application settings.

    Returns:
        The newly created model version number, as a string.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="register-mlp") as run:
        mlflow.pytorch.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
        logger.info(
            "Registered '%s' as version %s", REGISTERED_MODEL_NAME, result.version
        )
        return result.version


def promote_through_lifecycle(version: str, settings: Settings) -> None:
    """Promote a model version through Staging and Production aliases.

    Args:
        version: Model version number to promote.
        settings: Application settings.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="staging", version=version
    )
    logger.info("Version %s promoted to @staging", version)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="production", version=version
    )
    logger.info("Version %s promoted to @production", version)

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            "MLP embedding recommender trained on MovieLens 100K. "
            "RMSE=0.273 on test set, outperforming SVD and Popularity "
            "baselines by 2.6x on rating prediction. See Model Card "
            "for ranking metric trade-offs and known limitations."
        ),
    )


def main() -> None:
    """Register the trained MLP and promote it to production."""
    settings = Settings()
    model = load_trained_model(settings)

    version = log_and_register(model, settings)
    promote_through_lifecycle(version, settings)

    print("\n" + "=" * 60)
    print("MODEL REGISTRY — REGISTRATION COMPLETE")
    print("=" * 60)
    print(f"  Model name : {REGISTERED_MODEL_NAME}")
    print(f"  Version    : {version}")
    print("  Aliases    : @staging, @production")
    print(
        f"  View at    : {settings.mlflow_tracking_uri}/#/models/"
        f"{REGISTERED_MODEL_NAME}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
