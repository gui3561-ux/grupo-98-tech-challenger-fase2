"""Testes de fumaça para a Factory e as estratégias de recomendação."""

import pytest

from ecommerce_recommender.models import (
    PopularityRecommender,
    Recommender,
    RecommenderFactory,
)


def test_factory_cria_popularity() -> None:
    """A factory deve instanciar a estratégia de popularidade."""
    rec = RecommenderFactory.create("popularity", train_items=[1, 1, 2, 3])
    assert isinstance(rec, PopularityRecommender)
    assert isinstance(rec, Recommender)


def test_factory_tipo_desconhecido() -> None:
    """A factory deve levantar ValueError para tipos desconhecidos."""
    with pytest.raises(ValueError):
        RecommenderFactory.create("inexistente")


def test_popularity_recommend_ordena_por_frequencia() -> None:
    """O item mais frequente deve ser recomendado primeiro."""
    rec = RecommenderFactory.create("popularity", train_items=[1, 1, 1, 2, 2, 3])
    assert rec.recommend(user_id=0, top_k=2) == [1, 2]
