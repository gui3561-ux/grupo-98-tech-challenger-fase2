"""
Baseline recommendation models for comparison against the MLP.

These baselines do not require gradient-based training, but are wrapped as
nn.Module to stay compatible with ModelFactory, which registers and instantiates
models under a single type contract.
"""

from collections import Counter

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from torch import Tensor, nn

from models.factory import ModelFactory


@ModelFactory.register("popularity")
class PopularityRecommender(nn.Module):
    """
    Non-personalised baseline - ranks items by training frequency.

    fit() must be called once with training interactions before
    forward() is used for scoring.
    """

    def __init__(self, num_users: int, num_items: int) -> None:
        """
        Initialize the popularity baseline.

        Args:
            num_users: Total number of unique users (unused,
              kept for interface compatibility with ModelFactory.create()).
            num_items: Total number of unique items.
        """
        super().__init__()
        self.num_items = num_items
        self._item_scores = torch.zeros(num_items)
        self._is_fitted = False

    def fit(self, item_ids: np.ndarray) -> None:
        """
        Compute normalised popularity score per item.

        Args:
            item_ids: Array of item indices observed in training.
        """
        counts = Counter(item_ids.tolist())
        total = sum(counts.values()) or 1
        scores = torch.zeros(self.num_items)
        for item, count in counts.items():
            scores[item] = count / total
        self._item_scores = scores
        self._is_fitted = True

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Returns the popularity score for each item, ignoring the user.

        Args:
            user_ids: Tensor of user indices (ignored, kept for API
                compatibiliy with EmbeddingMLP.forward()).
            item_ids: Tensor of item indices to score.

        Returns:
            Tensor of popularity scores in [0, 1].
        """
        if not self._is_fitted:
            msg = "PopularityRecommender must be fitted before predicting."
            raise RuntimeError(msg)
        return self._item_scores[item_ids.cpu()]


@ModelFactory.register("svd")
class SVDRecommender(nn.Module):
    """
    Matrix factorisation baseline using Sckit-Learn TruncatedSVD.

    fit() muste be called once with training interactions before
    forward() is used for scoring.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_components: int = 50,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the SVD baseline.

        Args:
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            n_components: Number of latent factors.
            random_state: Seed for reproducibility.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._user_factors: np.ndarray = np.array([])
        self._item_factors: np.ndarray = np.array([])
        self._is_fitted = False

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        """Build the user-item matrix and decompose it with SVD.

        Args:
            user_ids: Array of user indices.
            item_ids: Array of item indices.
            ratings: Array of rating values (same length as ids).
        """
        matrix = csr_matrix(
            (ratings, (user_ids, item_ids)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32,
        )
        self._user_factors = normalize(self._svd.fit_transform(matrix))
        self._item_factors = normalize(self._svd.components_.T)
        self._is_fitted = True

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Score user-item pairs via dot product of latent factors.

        Args:
            user_ids: Tensor of user indices.
            item_ids: Tensor of item indices.

        Returns:
            Tensor of similarity scores.
        """
        if not self._is_fitted:
            msg = "SVDRecommender must be fitted before predicting."
            raise RuntimeError(msg)
        u = user_ids.cpu().numpy()
        i = item_ids.cpu().numpy()
        scores = np.sum(self._user_factors[u] * self._item_factors[i], axis=1)
        return torch.from_numpy(scores).float()
