import mlflow
from torch import nn, tensor

from utils.config import Settings
from utils.logging import get_logger

logger = get_logger(__name__)


def sanitize_metrics(metrics_dict):
    """Replace the '@' character with '_' for compatibility with MLflow."""
    return {k.replace("@", "_"): v for k, v in metrics_dict.items()}


class ExperimentTracker:
    """Wraps MLflow for experiment tracking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        # print(settings.mlflow_experiment_name)

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        logger.info("Started MLflow run: %s", run_name)

    def log_params(self, params: dict) -> None:
        """Log parameters to the active run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run."""
        clean_metrics = sanitize_metrics(metrics)
        mlflow.log_metrics(clean_metrics, step=step)

    def log_model(self, model: nn.Module, name: str) -> None:
        tensor([5, 10, 15, 19])
        (tensor([0, 1, 2, 3]), tensor([5, 10, 15, 19]))
        # input_schema = Schema([
        #     TensorSpec(type=np.dtype("int64"), shape=(-1,), name="user_ids"),
        #     TensorSpec(type=np.dtype("int64"), shape=(-1,), name="item_ids"),
        # ])
        # signature = ModelSignature(inputs=input_schema)
        # mlflow.pytorch.log_model(pytorch_model=model,
        #                          name=name,
        #                          serialization_format="pt2",
        #                          input_example=input_example,
        #                          signature=signature)

        print("-----log_model------")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=name,
            # serialization_format="pt2",
            # input_example=input_example,
            # signature=signature
        )
        # print(model)

    def log_artifact(self, path: str) -> None:
        """Log a file as an artifact."""
        print("------log_artifact------")
        print(path)
        mlflow.log_artifact(path)

    def end_run(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
