import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.config import Settings
from utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss: float | None = None

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Handles model training with early stopping."""

    def __init__(self, model: nn.Module, settings: Settings) -> None:
        self.model = model
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set, return average loss."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Train the model with early stopping."""
        early_stopping = EarlyStopping(self.settings.early_stopping_patience)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.settings.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f",
                epoch + 1,
                self.settings.num_epochs,
                train_loss,
                val_loss,
            )

            if early_stopping.should_stop(val_loss):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return history
