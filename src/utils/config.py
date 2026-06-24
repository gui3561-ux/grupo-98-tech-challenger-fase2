from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    random_seed: int = 42

    mlflow_tracking_uri: str = "http://0.0.0.0:5000"
    mlflow_experiment_name: str = "ecommerce-recommender"

    num_users: int = 1502
    num_items: int = 2598
    min_interactions: int = 10

    batch_size: int = 256
    embedding_dim: int = 64
    hidden_dims: str = "128,64,32"
    num_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 5

    data_raw_path: Path = Path("data/raw/ecommerce")
    data_processed_path: Path = Path("data/processed")
    data_features_path: Path = Path("data/features")
    models_path: Path = Path("models")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "http://0.0.0.1:5173,http://0.0.0.1:3000"

    @property
    def hidden_dims_list(self) -> list[int]:
        """Parse comma-separated hidden dimensions string."""
        return [int(d) for d in self.hidden_dims.split(",")]
