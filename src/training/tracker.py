import mlflow
from torch import nn

from utils.config import Settings
from utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Wraps MLflow for experiment tracking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        logger.info("Started MLflow run: %s", run_name)

    def log_params(self, params: dict) -> None:
        """Log parameters to the active run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, name: str) -> None:
        """Log a PyTorch model artifact."""
        mlflow.pytorch.log_model(model, name)

    def log_artifact(self, path: str) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(path)

    def end_run(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
