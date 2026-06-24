import numpy as np
import torch
from torch import Tensor


def rmse(predictions: Tensor, targets: Tensor) -> float:
    """Compute Root Mean Squared Error."""
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def precision_at_k(
    predicted_items: list[int], relevant_items: set[int], k: int
) -> float:
    """Compute Precision@K for a single user."""
    top_k = predicted_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / k


def recall_at_k(predicted_items: list[int], relevant_items: set[int], k: int) -> float:
    """Compute Recall@K for a single user."""
    if not relevant_items:
        return 0.0
    top_k = predicted_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / len(relevant_items)


def ndcg_at_k(predicted_items: list[int], relevant_items: set[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain@K."""
    top_k = predicted_items[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant_items
    )
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg
