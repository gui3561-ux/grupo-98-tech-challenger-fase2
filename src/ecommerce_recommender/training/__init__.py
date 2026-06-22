"""Subpacote de treino: configuração, loop de épocas e early stopping."""

from .trainer import TrainConfig, TrainResult, train_epoch, train_model

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train_epoch",
    "train_model",
]
