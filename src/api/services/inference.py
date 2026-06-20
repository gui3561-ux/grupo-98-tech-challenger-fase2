import torch

from models.factory import ModelFactory
from models.mlp import MLPRecommender  # noqa: F401
from preprocessing.feature_engineering import load_encoders
from utils import Settings, get_logger

logger = get_logger(__name__)


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: torch.nn.Module | None = None
        self.encoders: dict | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.encoders is not None

    def load_model(self) -> None:
        encoders_path = self.settings.data_features_path / "encoders.json"
        model_path = self.settings.models_path / "model.pt"

        self.encoders = load_encoders(str(encoders_path))

        model = ModelFactory.create(
            "mlp",
            num_users=self.encoders["num_users"],
            num_items=self.encoders["num_items"],
            embedding_dim=self.settings.embedding_dim,
            hidden_dims=self.settings.hidden_dims_list,
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info("Model loaded successfully")

    def recommend(self, user_id: str, top_k: int = 10) -> list[dict] | None:
        if not self.is_ready:
            return None

        user_encoder = self.encoders["user_encoder"]
        item_encoder = self.encoders["item_encoder"]

        if user_id not in user_encoder:
            return None

        user_idx = user_encoder[user_id]
        num_items = self.encoders["num_items"]
        item_decoder = {v: k for k, v in item_encoder.items()}

        with torch.no_grad():
            user_tensor = torch.full(
                (num_items,), user_idx, dtype=torch.long, device=self.device
            )
            item_tensor = torch.arange(num_items, dtype=torch.long, device=self.device)
            scores = self.model(user_tensor, item_tensor)

        top_indices = torch.argsort(scores, descending=True)[:top_k].cpu().tolist()

        results = []
        for rank, item_idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "item_id": item_decoder[item_idx],
                    "score": round(scores[item_idx].item(), 4),
                    "rank": rank,
                }
            )
        return results
