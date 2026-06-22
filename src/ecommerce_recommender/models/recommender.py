"""Modelos de recomendação e seus padrões de criação.

Aplica os design patterns:
    * Strategy: :class:`Recommender` define a interface comum; cada modelo
      (MLP, popularidade, SVD) é uma estratégia intercambiável.
    * Factory: :class:`RecommenderFactory` centraliza a criação das estratégias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from torch import nn


class EmbeddingMLP(nn.Module):
    """MLP baseada em embeddings para previsão de score usuário–item.

    Arquitetura:
        user_embedding + item_embedding -> concat -> MLP -> Sigmoid -> score.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
    ) -> None:
        """Constrói as tabelas de embedding e a cabeça MLP.

        Args:
            num_users: Tamanho do vocabulário de usuários.
            num_items: Tamanho do vocabulário de itens.
            emb_dim: Dimensão dos vetores de embedding.
            hidden_dims: Tamanhos das camadas ocultas.
            dropout: Taxa de dropout após cada camada oculta.
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.mlp = self._build_mlp(emb_dim * 2, hidden_dims, dropout)

    @staticmethod
    def _build_mlp(
        in_dim: int, hidden_dims: list[int], dropout: float
    ) -> nn.Sequential:
        """Monta a sequência de camadas densas do MLP.

        Args:
            in_dim: Dimensão de entrada (embeddings concatenados).
            hidden_dims: Tamanhos das camadas ocultas.
            dropout: Taxa de dropout após cada camada oculta.

        Returns:
            Bloco sequencial terminando em Linear(…, 1) + Sigmoid.
        """
        layers: list[nn.Module] = []
        for hidden in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        return nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass pelas embeddings e pelo MLP.

        Args:
            user_ids: Tensor long de formato (B,) com índices de usuários.
            item_ids: Tensor long de formato (B,) com índices de itens.

        Returns:
            Tensor float de formato (B,) com scores em (0, 1).
        """
        user_vec = self.user_emb(user_ids)
        item_vec = self.item_emb(item_ids)
        return self.mlp(torch.cat([user_vec, item_vec], dim=-1)).squeeze(-1)


class Recommender(ABC):
    """Interface (Strategy) comum a todos os recomendadores."""

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Estima o score de relevância de um par usuário–item.

        Args:
            user_id: Índice do usuário.
            item_id: Índice do item.

        Returns:
            Score de relevância previsto.
        """

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Gera a lista ordenada de itens recomendados para um usuário.

        Args:
            user_id: Índice do usuário.
            top_k: Quantidade de itens a recomendar.

        Returns:
            Lista de índices de itens, ordenada por relevância decrescente.
        """


class MLPRecommender(Recommender):
    """Estratégia de recomendação baseada na :class:`EmbeddingMLP`."""

    def __init__(
        self, model: EmbeddingMLP, num_items: int, device: str = "cpu"
    ) -> None:
        """Inicializa o recomendador com um modelo treinado.

        Args:
            model: Instância (já treinada) de :class:`EmbeddingMLP`.
            num_items: Número total de itens do catálogo.
            device: Dispositivo de inferência ("cpu" ou "cuda").
        """
        self._model = model.to(device)
        self._num_items = num_items
        self._device = device

    def predict(self, user_id: int, item_id: int) -> float:
        """Estima o score de relevância de um par usuário–item.

        Args:
            user_id: Índice do usuário.
            item_id: Índice do item.

        Returns:
            Score de relevância previsto em (0, 1).
        """
        self._model.eval()
        with torch.no_grad():
            user = torch.tensor([user_id], dtype=torch.long, device=self._device)
            item = torch.tensor([item_id], dtype=torch.long, device=self._device)
            return float(self._model(user, item).item())

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Recomenda os top-K itens com maior score para o usuário.

        Args:
            user_id: Índice do usuário.
            top_k: Quantidade de itens a recomendar.

        Returns:
            Lista de índices de itens ordenada por score decrescente.
        """
        self._model.eval()
        items = torch.arange(self._num_items, dtype=torch.long, device=self._device)
        users = torch.full_like(items, user_id)
        with torch.no_grad():
            scores = self._model(users, items).cpu().numpy()
        return np.argsort(scores)[::-1][:top_k].tolist()


class PopularityRecommender(Recommender):
    """Estratégia baseline que recomenda os itens mais populares."""

    def __init__(self, train_items: list[int]) -> None:
        """Indexa a popularidade dos itens a partir do treino.

        Args:
            train_items: Lista de índices de itens observados no treino.
        """
        counts = Counter(train_items)
        total = sum(counts.values()) or 1
        self._scores = {item: count / total for item, count in counts.items()}
        self._ranked = [item for item, _ in counts.most_common()]

    def predict(self, user_id: int, item_id: int) -> float:
        """Retorna a popularidade global do item (ignora o usuário).

        Args:
            user_id: Índice do usuário (não utilizado).
            item_id: Índice do item.

        Returns:
            Frequência relativa do item no conjunto de treino.
        """
        return self._scores.get(item_id, 0.0)

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Recomenda os itens globalmente mais populares.

        Args:
            user_id: Índice do usuário (não utilizado).
            top_k: Quantidade de itens a recomendar.

        Returns:
            Lista dos top-K itens mais populares.
        """
        return self._ranked[:top_k]


class SVDRecommender(Recommender):
    """Estratégia baseline de fatoração de matrizes (SVD)."""

    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray) -> None:
        """Inicializa o recomendador com os fatores latentes.

        Args:
            user_factors: Matriz (num_users, k) de fatores de usuário.
            item_factors: Matriz (num_items, k) de fatores de item.
        """
        self._user_factors = user_factors
        self._item_factors = item_factors

    def predict(self, user_id: int, item_id: int) -> float:
        """Estima o score como produto interno dos fatores latentes.

        Args:
            user_id: Índice do usuário.
            item_id: Índice do item.

        Returns:
            Score previsto, ou 0.0 se os índices estiverem fora do intervalo.
        """
        if user_id >= self._user_factors.shape[0]:
            return 0.0
        if item_id >= self._item_factors.shape[0]:
            return 0.0
        return float(np.dot(self._user_factors[user_id], self._item_factors[item_id]))

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Recomenda os top-K itens com maior score latente.

        Args:
            user_id: Índice do usuário.
            top_k: Quantidade de itens a recomendar.

        Returns:
            Lista de índices de itens ordenada por score decrescente.
        """
        scores = self._user_factors[user_id] @ self._item_factors.T
        return np.argsort(scores)[::-1][:top_k].tolist()


class RecommenderFactory:
    """Factory que cria recomendadores a partir de um identificador."""

    _BUILDERS = {
        "mlp": MLPRecommender,
        "popularity": PopularityRecommender,
        "svd": SVDRecommender,
    }

    @classmethod
    def create(cls, kind: str, **kwargs: object) -> Recommender:
        """Cria a estratégia de recomendação solicitada.

        Args:
            kind: Tipo do recomendador ("mlp", "popularity" ou "svd").
            **kwargs: Argumentos repassados ao construtor da estratégia.

        Returns:
            Instância concreta de :class:`Recommender`.

        Raises:
            ValueError: Se ``kind`` não corresponder a um modelo conhecido.
        """
        try:
            builder = cls._BUILDERS[kind]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._BUILDERS))
            raise ValueError(
                f"Recomendador desconhecido: {kind!r}. Opções: {valid}."
            ) from exc
        return builder(**kwargs)
