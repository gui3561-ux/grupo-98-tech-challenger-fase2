import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """PyTorch Dataset for user-item interaction data."""

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
