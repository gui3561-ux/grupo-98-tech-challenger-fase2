import numpy as np
import pandas as pd
import pytest

from utils.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Test settings with overridden values."""
    return Settings(
        app_env="test",
        random_seed=42,
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="test-experiment",
        num_users=10,
        num_items=20,
        min_interactions=2,
        batch_size=4,
        embedding_dim=8,
        hidden_dims="16,8",
        num_epochs=2,
        learning_rate=0.01,
        early_stopping_patience=2,
    )


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Sample events DataFrame for testing preprocessing."""
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 2, 3, 1, 2],
            "itemid": [10, 10, 20, 10, 20, 30, 10, 20, 30, 40, 30, 40, 40, 40, 10],
            "event": [
                "view",
                "addtocart",
                "view",
                "view",
                "transaction",
                "view",
                "view",
                "addtocart",
                "view",
                "view",
                "view",
                "view",
                "transaction",
                "view",
                "view",
            ],
            "timestamp": list(range(15)),
        }
    )


@pytest.fixture
def sample_interactions_df() -> pd.DataFrame:
    """Sample processed interactions DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "visitorid": np.random.randint(0, 5, n),
            "itemid": np.random.randint(0, 10, n),
            "rating": np.random.uniform(0, 1, n),
            "timestamp": list(range(n)),
        }
    )
