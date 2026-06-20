from utils.config import Settings


class TestSettings:
    def test_default_values(self) -> None:
        settings = Settings(
            _env_file=None,
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="test",
        )
        assert settings.random_seed == 42
        assert settings.batch_size == 256
        assert settings.embedding_dim == 64

    def test_hidden_dims_list(self) -> None:
        settings = Settings(
            _env_file=None,
            hidden_dims="128,64,32",
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="test",
        )
        assert settings.hidden_dims_list == [128, 64, 32]

    def test_custom_hidden_dims(self) -> None:
        settings = Settings(
            _env_file=None,
            hidden_dims="256,128",
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="test",
        )
        assert settings.hidden_dims_list == [256, 128]
