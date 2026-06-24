import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates a recommendation model on test data."""

    def __init__(self, model: nn.Module, top_k: int = 10) -> None:
        self.model = model
        self.top_k = top_k
        self.device = next(model.parameters()).device

    def compute_rmse(self, dataloader: DataLoader) -> float:
        """Compute RMSE over all predictions."""
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                preds = self.model(user_ids, item_ids)
                all_preds.append(preds.cpu())
                all_targets.append(ratings)

        predictions = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        return rmse(predictions, targets)

    def compute_ranking_metrics(
        self,
        test_user_items: dict[int, set[int]],
        num_items: int,
    ) -> dict[str, float]:
        """Compute Precision@K, Recall@K, NDCG@K averaged over users."""
        self.model.eval()
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []

        with torch.no_grad():
            for user_id, relevant_items in test_user_items.items():
                user_tensor = torch.full((num_items,), user_id, dtype=torch.long).to(
                    self.device
                )
                item_tensor = torch.arange(num_items).to(self.device)
                scores = self.model(user_tensor, item_tensor)
                ranked_items = torch.argsort(scores, descending=True).cpu().tolist()

                precisions.append(
                    precision_at_k(ranked_items, relevant_items, self.top_k)
                )
                recalls.append(recall_at_k(ranked_items, relevant_items, self.top_k))
                ndcgs.append(ndcg_at_k(ranked_items, relevant_items, self.top_k))

        return {
            f"precision@{self.top_k}": sum(precisions) / len(precisions),
            f"recall@{self.top_k}": sum(recalls) / len(recalls),
            f"ndcg@{self.top_k}": sum(ndcgs) / len(ndcgs),
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        test_user_items: dict[int, set[int]],
        num_items: int,
    ) -> dict[str, float]:
        """Run full evaluation: RMSE + ranking metrics."""
        metrics = {"rmse": self.compute_rmse(dataloader)}
        ranking = self.compute_ranking_metrics(test_user_items, num_items)
        metrics.update(ranking)
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
