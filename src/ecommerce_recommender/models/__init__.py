"""Subpacote de modelos: rede neural, baselines e fábrica (Factory/Strategy)."""

from .recommender import (
    EmbeddingMLP,
    MLPRecommender,
    PopularityRecommender,
    Recommender,
    RecommenderFactory,
    SVDRecommender,
)

__all__ = [
    "EmbeddingMLP",
    "Recommender",
    "MLPRecommender",
    "PopularityRecommender",
    "SVDRecommender",
    "RecommenderFactory",
]
