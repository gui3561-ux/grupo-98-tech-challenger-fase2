import pytest
import torch

from models.factory import ModelFactory
from models.mlp import MLPRecommender


class TestModelFactory:
    def test_create_mlp(self) -> None:
        model = ModelFactory.create(
            "mlp",
            num_users=10,
            num_items=20,
            embedding_dim=8,
            hidden_dims=[16, 8],
        )
        assert isinstance(model, MLPRecommender)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            ModelFactory.create("unknown_model")

    def test_available_models(self) -> None:
        models = ModelFactory.available_models()
        assert "mlp" in models


class TestMLPRecommender:
    def test_forward_shape(self) -> None:
        model = MLPRecommender(
            num_users=10,
            num_items=20,
            embedding_dim=8,
            hidden_dims=[16, 8],
        )
        user_ids = torch.tensor([0, 1, 2, 3])
        item_ids = torch.tensor([5, 10, 15, 19])
        output = model(user_ids, item_ids)
        assert output.shape == (4,)

    def test_output_range(self) -> None:
        model = MLPRecommender(
            num_users=10,
            num_items=20,
            embedding_dim=8,
            hidden_dims=[16, 8],
        )
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        output = model(user_ids, item_ids)
        assert (output >= 0).all()
        assert (output <= 1).all()
