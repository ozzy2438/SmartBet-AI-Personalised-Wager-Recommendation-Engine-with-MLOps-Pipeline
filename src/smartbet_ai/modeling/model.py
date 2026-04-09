"""Two-tower neural recommender model and checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from smartbet_ai.common.constants import MARKET_FEATURE_COLUMNS, USER_FEATURE_COLUMNS


class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""

    def __init__(
        self,
        num_users: int,
        num_sports: int,
        embedding_dim: int = 64,
        n_features: int = len(USER_FEATURE_COLUMNS) - 1,  # sport_encoded excluded (embedded separately)
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        final_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.sport_embedding = nn.Embedding(num_sports, 16)

        layers: list[nn.Module] = []
        prev_dim = embedding_dim + 16 + n_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, final_embedding_dim)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embedding(user_ids)
        sport_indices = user_features[:, 0].long()
        sport_embedding = self.sport_embedding(sport_indices)
        # Exclude sport_encoded (col 0) — already captured by sport_embedding
        dense_features = user_features[:, 1:]
        x = torch.cat([user_embedding, sport_embedding, dense_features], dim=-1)
        x = self.network(x)
        x = self.output_layer(x)
        return F.normalize(x, p=2, dim=-1)


class MarketTower(nn.Module):
    """Encodes market features into the same embedding space as users."""

    def __init__(
        self,
        num_markets: int,
        num_sports: int,
        num_market_types: int,
        embedding_dim: int = 64,
        n_features: int = len(MARKET_FEATURE_COLUMNS) - 2,  # sport_encoded + market_type_encoded excluded (embedded separately)
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        final_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        self.market_embedding = nn.Embedding(num_markets, embedding_dim)
        self.sport_embedding = nn.Embedding(num_sports, 16)
        self.market_type_embedding = nn.Embedding(num_market_types, 8)

        layers: list[nn.Module] = []
        prev_dim = embedding_dim + 16 + 8 + n_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, final_embedding_dim)

    def forward(self, market_ids: torch.Tensor, market_features: torch.Tensor) -> torch.Tensor:
        market_embedding = self.market_embedding(market_ids)
        sport_indices = market_features[:, 0].long()
        market_type_indices = market_features[:, 1].long()

        sport_embedding = self.sport_embedding(sport_indices)
        market_type_embedding = self.market_type_embedding(market_type_indices)

        # Exclude sport_encoded (col 0) and market_type_encoded (col 1) — already embedded
        dense_features = market_features[:, 2:]
        x = torch.cat([market_embedding, sport_embedding, market_type_embedding, dense_features], dim=-1)
        x = self.network(x)
        x = self.output_layer(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerRecommender(nn.Module):
    """Scores user-market compatibility in a shared embedding space."""

    def __init__(self, user_tower: UserTower, market_tower: MarketTower, temperature: float = 0.1) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.market_tower = market_tower
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def _temperature(self) -> torch.Tensor:
        return torch.clamp(self.temperature, min=1e-3)

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        market_ids: torch.Tensor,
        market_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embedding = self.user_tower(user_ids, user_features)
        market_embedding = self.market_tower(market_ids, market_features)
        logits = (user_embedding * market_embedding).sum(dim=-1) / self._temperature()
        return logits, user_embedding, market_embedding

    def get_user_embedding(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids, user_features)

    def get_market_embedding(self, market_ids: torch.Tensor, market_features: torch.Tensor) -> torch.Tensor:
        return self.market_tower(market_ids, market_features)

    def score_candidates(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        market_ids: torch.Tensor,
        market_features: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding = self.get_user_embedding(user_ids, user_features)
        market_embedding = self.get_market_embedding(market_ids, market_features)
        return torch.matmul(user_embedding, market_embedding.T) / self._temperature()

    def recommend(
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor,
        all_market_ids: torch.Tensor,
        all_market_features: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            scores = self.score_candidates(user_id, user_features, all_market_ids, all_market_features).squeeze(0)
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
            return top_indices, top_scores


def build_model_from_config(config: dict) -> TwoTowerRecommender:
    """Construct a model from the unified configuration dictionary."""
    model_config = config["model"]
    data_config = config["data"]

    user_tower = UserTower(
        num_users=data_config["n_users"],
        num_sports=len(data_config["sports"]),
        embedding_dim=model_config["embedding_dim"],
        hidden_dims=model_config["user_tower"]["hidden_dims"],
        dropout=model_config["user_tower"]["dropout"],
        final_embedding_dim=model_config["final_embedding_dim"],
    )
    market_tower = MarketTower(
        num_markets=data_config["n_markets"],
        num_sports=len(data_config["sports"]),
        num_market_types=len(data_config["market_types"]),
        embedding_dim=model_config["embedding_dim"],
        hidden_dims=model_config["market_tower"]["hidden_dims"],
        dropout=model_config["market_tower"]["dropout"],
        final_embedding_dim=model_config["final_embedding_dim"],
    )
    model = TwoTowerRecommender(
        user_tower=user_tower,
        market_tower=market_tower,
        temperature=model_config["temperature"],
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return model


def load_model_from_checkpoint(model_path: Path | str, device: torch.device | str | None = None) -> tuple[TwoTowerRecommender, dict]:
    """Load a trained model checkpoint and return the model plus saved config."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=target_device, weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(target_device)
    model.eval()
    return model, checkpoint
