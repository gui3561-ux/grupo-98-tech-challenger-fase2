"""Testes de fumaça para as métricas de avaliação."""

from ecommerce_recommender.evaluation import (
    ndcg_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_perfeito() -> None:
    """Precision@K deve ser 1.0 quando todas as top-K são relevantes."""
    assert precision_at_k({1, 2, 3}, [1, 2, 3], k=3) == 1.0


def test_recall_at_k_parcial() -> None:
    """Recall@K reflete a fração de relevantes recuperados."""
    assert recall_at_k({1, 2, 3, 4}, [1, 2], k=2) == 0.5


def test_ndcg_premia_posicao() -> None:
    """nDCG deve ser maior quando o item relevante vem antes no ranking."""
    antes = ndcg_at_k({1}, [1, 9, 8], k=3)
    depois = ndcg_at_k({1}, [9, 8, 1], k=3)
    assert antes > depois


def test_pr_auc_separavel() -> None:
    """PR-AUC deve ser 1.0 para scores perfeitamente separáveis."""
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
