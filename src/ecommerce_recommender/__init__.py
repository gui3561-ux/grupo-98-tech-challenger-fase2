"""Pacote do sistema de recomendação de produtos (Tech Challenge Fase 02)."""

from .config import Settings, get_settings
from .evaluation import (
    f1_at_threshold,
    hit_rate_at_k,
    ndcg_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from .models import (
    EmbeddingMLP,
    MLPRecommender,
    PopularityRecommender,
    Recommender,
    RecommenderFactory,
    SVDRecommender,
)
from .training import TrainConfig, TrainResult, train_epoch, train_model

__all__ = [
    "Settings",
    "get_settings",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "pr_auc",
    "roc_auc",
    "f1_at_threshold",
    "EmbeddingMLP",
    "Recommender",
    "MLPRecommender",
    "PopularityRecommender",
    "SVDRecommender",
    "RecommenderFactory",
    "TrainConfig",
    "TrainResult",
    "train_epoch",
    "train_model",
]
