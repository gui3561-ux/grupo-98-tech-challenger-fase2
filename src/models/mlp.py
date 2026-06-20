import torch
from torch import Tensor, nn

from models.factory import ModelFactory


@ModelFactory.register("mlp")
class MLPRecommender(nn.Module):
    """MLP-based recommendation model with user/item embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Predict interaction score for user-item pairs."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(combined).squeeze(1)
