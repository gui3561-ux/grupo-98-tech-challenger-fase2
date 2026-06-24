from utils.config import Settings
from ecommerce_recommender.config import Settings, get_settings

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


# """Testes de fumaça para a configuração via Pydantic Settings."""
# def test_get_settings_retorna_instancia() -> None:
#     """get_settings deve retornar uma instância de Settings."""
#     assert isinstance(get_settings(), Settings)


# def test_hidden_dims_list_parseia_string() -> None:
#     """hidden_dims deve ser convertido em lista de inteiros."""
#     settings = Settings(hidden_dims="128,64,32")
#     assert settings.hidden_dims_list() == [128, 64, 32]


# def test_random_seed_default() -> None:
#     """A semente padrão deve ser 42 para reprodutibilidade."""
#     assert Settings().random_seed == 42
