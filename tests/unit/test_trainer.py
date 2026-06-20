from torch.utils.data import DataLoader

from models.factory import ModelFactory
from training.dataset import InteractionDataset
from training.trainer import EarlyStopping, Trainer
from utils.config import Settings


class TestInteractionDataset:
    def test_length(self) -> None:
        import numpy as np

        dataset = InteractionDataset(
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
            np.array([0.5, 0.8, 1.0]),
        )
        assert len(dataset) == 3

    def test_getitem(self) -> None:
        import numpy as np

        dataset = InteractionDataset(
            np.array([0, 1]),
            np.array([3, 4]),
            np.array([0.5, 0.8]),
        )
        user, item, rating = dataset[0]
        assert user.item() == 0
        assert item.item() == 3
        assert abs(rating.item() - 0.5) < 1e-6


class TestEarlyStopping:
    def test_does_not_stop_when_improving(self) -> None:
        es = EarlyStopping(patience=3)
        assert not es.should_stop(1.0)
        assert not es.should_stop(0.9)
        assert not es.should_stop(0.8)

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=2)
        es.should_stop(1.0)
        es.should_stop(1.1)
        assert es.should_stop(1.2)


class TestTrainer:
    def test_train_one_epoch(self, settings: Settings) -> None:
        import numpy as np

        model = ModelFactory.create(
            "mlp",
            num_users=settings.num_users,
            num_items=settings.num_items,
            embedding_dim=settings.embedding_dim,
            hidden_dims=settings.hidden_dims_list,
        )
        dataset = InteractionDataset(
            np.random.randint(0, settings.num_users, 20),
            np.random.randint(0, settings.num_items, 20),
            np.random.uniform(0, 1, 20).astype(np.float32),
        )
        loader = DataLoader(dataset, batch_size=settings.batch_size)
        trainer = Trainer(model, settings)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss > 0
