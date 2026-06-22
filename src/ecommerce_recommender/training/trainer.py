"""Rotinas de treino da rede neural com early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..models import EmbeddingMLP


@dataclass
class TrainConfig:
    """Hiperparâmetros de treino.

    Attributes:
        num_epochs: Número máximo de épocas.
        batch_size: Tamanho do mini-batch.
        learning_rate: Taxa de aprendizado do otimizador Adam.
        weight_decay: Regularização L2 do otimizador.
        patience: Épocas sem melhora antes do early stopping.
        max_grad_norm: Limite de norma para clipping de gradiente.
        device: Dispositivo de treino ("cpu" ou "cuda").
    """

    num_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 7
    max_grad_norm: float = 1.0
    device: str = "cpu"


@dataclass
class TrainResult:
    """Resultado de um treino.

    Attributes:
        best_val_loss: Menor loss de validação observada.
        train_losses: Histórico de loss de treino por época.
        val_losses: Histórico de loss de validação por época.
        epochs_run: Número de épocas efetivamente executadas.
    """

    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    epochs_run: int = 0


def train_epoch(
    model: EmbeddingMLP,
    users: torch.Tensor,
    items: torch.Tensor,
    ratings: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
) -> float:
    """Executa uma época completa de treino e retorna a loss média.

    Args:
        model: Instância de :class:`EmbeddingMLP`.
        users: Tensor de índices de usuários.
        items: Tensor de índices de itens.
        ratings: Tensor de ratings normalizados (alvo).
        optimizer: Otimizador a ser aplicado.
        criterion: Função de perda (ex.: MSELoss).
        config: Hiperparâmetros de treino.

    Returns:
        Loss média ao longo de todos os mini-batches.
    """
    model.train()
    perm = torch.randperm(len(users))
    users, items, ratings = users[perm], items[perm], ratings[perm]

    total_loss = 0.0
    n_batches = max(1, len(users) // config.batch_size)
    for batch in range(n_batches):
        start, end = batch * config.batch_size, (batch + 1) * config.batch_size
        loss = _train_step(
            model, users[start:end], items[start:end], ratings[start:end],
            optimizer, criterion, config,
        )
        total_loss += loss
    return total_loss / n_batches


def _train_step(
    model: EmbeddingMLP,
    users: torch.Tensor,
    items: torch.Tensor,
    ratings: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
) -> float:
    """Executa um passo de otimização sobre um mini-batch.

    Args:
        model: Instância de :class:`EmbeddingMLP`.
        users: Sub-tensor de índices de usuários do batch.
        items: Sub-tensor de índices de itens do batch.
        ratings: Sub-tensor de ratings normalizados do batch.
        optimizer: Otimizador a ser aplicado.
        criterion: Função de perda.
        config: Hiperparâmetros de treino.

    Returns:
        Valor escalar da loss do batch.
    """
    user_batch = users.to(config.device)
    item_batch = items.to(config.device)
    rating_batch = ratings.to(config.device)

    optimizer.zero_grad()
    loss = criterion(model(user_batch, item_batch), rating_batch)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def _validate(
    model: EmbeddingMLP,
    users: torch.Tensor,
    items: torch.Tensor,
    ratings: torch.Tensor,
    criterion: nn.Module,
    device: str,
) -> float:
    """Calcula a loss de validação sem atualizar os pesos.

    Args:
        model: Instância de :class:`EmbeddingMLP`.
        users: Tensor de índices de usuários de validação.
        items: Tensor de índices de itens de validação.
        ratings: Tensor de ratings normalizados de validação.
        criterion: Função de perda.
        device: Dispositivo de inferência.

    Returns:
        Loss escalar de validação.
    """
    model.eval()
    with torch.no_grad():
        preds = model(users.to(device), items.to(device))
        return float(criterion(preds, ratings.to(device)).item())


def train_model(
    model: EmbeddingMLP,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: TrainConfig,
) -> TrainResult:
    """Treina o modelo com early stopping baseado na loss de validação.

    Args:
        model: Instância de :class:`EmbeddingMLP` a ser treinada.
        train_data: Tupla (users, items, ratings) de treino.
        val_data: Tupla (users, items, ratings) de validação.
        config: Hiperparâmetros de treino.

    Returns:
        :class:`TrainResult` com a melhor loss e os históricos de treino.
    """
    model = model.to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    result = TrainResult(best_val_loss=float("inf"))
    best_state: dict[str, torch.Tensor] = {}
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, *train_data, optimizer, criterion, config)
        val_loss = _validate(model, *val_data, criterion, config.device)
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.epochs_run = epoch + 1

        if val_loss < result.best_val_loss - 1e-4:
            result.best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return result
