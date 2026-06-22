"""Testes de fumaça para a configuração via Pydantic Settings."""

from ecommerce_recommender.config import Settings, get_settings


def test_get_settings_retorna_instancia() -> None:
    """get_settings deve retornar uma instância de Settings."""
    assert isinstance(get_settings(), Settings)


def test_hidden_dims_list_parseia_string() -> None:
    """hidden_dims deve ser convertido em lista de inteiros."""
    settings = Settings(hidden_dims="128,64,32")
    assert settings.hidden_dims_list() == [128, 64, 32]


def test_random_seed_default() -> None:
    """A semente padrão deve ser 42 para reprodutibilidade."""
    assert Settings().random_seed == 42
