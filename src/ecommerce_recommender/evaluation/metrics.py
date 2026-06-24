"""Métricas de avaliação para recomendação.

Inclui métricas de ranking (Precision@K, Recall@K, nDCG@K, HitRate@K) e
métricas de classificação binária de relevância (PR-AUC, AUC-ROC, F1),
conforme definido no ML Canvas do projeto.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def precision_at_k(
    relevant: set[int], recommended: Sequence[int], k: int = 10
) -> float:
    """Fração das top-K recomendações que são relevantes.

    Args:
        relevant: Conjunto de itens relevantes (ground-truth).
        recommended: Lista ordenada de itens recomendados.
        k: Posição de corte do ranking.

    Returns:
        Precision@K no intervalo [0, 1].
    """
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(relevant: set[int], recommended: Sequence[int], k: int = 10) -> float:
    """Fração dos itens relevantes encontrados nas top-K recomendações.

    Args:
        relevant: Conjunto de itens relevantes (ground-truth).
        recommended: Lista ordenada de itens recomendados.
        k: Posição de corte do ranking.

    Returns:
        Recall@K no intervalo [0, 1].
    """
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(relevant: set[int], recommended: Sequence[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        relevant: Conjunto de itens relevantes (ground-truth).
        recommended: Lista ordenada de itens recomendados.
        k: Posição de corte do ranking.

    Returns:
        nDCG@K no intervalo [0, 1], sensível à posição do acerto.
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(relevant: set[int], recommended: Sequence[int], k: int = 10) -> float:
    """Indica se há ao menos um item relevante nas top-K recomendações.

    Args:
        relevant: Conjunto de itens relevantes (ground-truth).
        recommended: Lista ordenada de itens recomendados.
        k: Posição de corte do ranking.

    Returns:
        1.0 se houver ao menos um acerto, caso contrário 0.0.
    """
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR-AUC (Average Precision) — métrica técnica primária do projeto.

    Adequada a dados desbalanceados (recomendação), pois foca na precisão
    e no recall da classe relevante (positiva e rara).

    Args:
        y_true: Rótulos binários de relevância (1 = relevante, 0 = não).
        y_score: Scores de relevância previstos pelo modelo.

    Returns:
        Average Precision no intervalo [0, 1].
    """
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """AUC-ROC — métrica técnica secundária (discriminação geral).

    Args:
        y_true: Rótulos binários de relevância (1 = relevante, 0 = não).
        y_score: Scores de relevância previstos pelo modelo.

    Returns:
        Área sob a curva ROC no intervalo [0, 1].
    """
    return float(roc_auc_score(y_true, y_score))


def f1_at_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5
) -> float:
    """F1-score no limiar de operação informado.

    Args:
        y_true: Rótulos binários de relevância (1 = relevante, 0 = não).
        y_score: Scores de relevância previstos pelo modelo.
        threshold: Limiar para binarizar ``y_score``.

    Returns:
        F1-score no intervalo [0, 1].
    """
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    return float(f1_score(y_true, y_pred, zero_division=0))
