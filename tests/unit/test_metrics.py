import torch

from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


class TestRmse:
    def test_perfect_predictions(self) -> None:
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        assert rmse(preds, targets) == 0.0

    def test_known_value(self) -> None:
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([2.0, 3.0])
        assert abs(rmse(preds, targets) - 1.0) < 1e-6


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        predicted = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(predicted, relevant, k=5) == 1.0

    def test_none_relevant(self) -> None:
        predicted = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        assert precision_at_k(predicted, relevant, k=5) == 0.0

    def test_partial(self) -> None:
        predicted = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        assert precision_at_k(predicted, relevant, k=5) == 0.4


class TestRecallAtK:
    def test_all_recalled(self) -> None:
        predicted = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        assert recall_at_k(predicted, relevant, k=5) == 1.0

    def test_none_recalled(self) -> None:
        predicted = [1, 2, 3]
        relevant = {4, 5}
        assert recall_at_k(predicted, relevant, k=3) == 0.0

    def test_empty_relevant(self) -> None:
        predicted = [1, 2, 3]
        relevant: set[int] = set()
        assert recall_at_k(predicted, relevant, k=3) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        predicted = [1, 2, 3]
        relevant = {1, 2, 3}
        assert abs(ndcg_at_k(predicted, relevant, k=3) - 1.0) < 1e-6

    def test_no_relevant(self) -> None:
        predicted = [1, 2, 3]
        relevant = {4, 5}
        assert ndcg_at_k(predicted, relevant, k=3) == 0.0
