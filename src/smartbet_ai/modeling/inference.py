"""Shared inference helpers for the API and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from smartbet_ai.common.constants import MARKET_FEATURE_COLUMNS, USER_FEATURE_COLUMNS
from smartbet_ai.common.paths import MODELS_DIR, PROCESSED_DATA_DIR
from smartbet_ai.modeling.model import TwoTowerRecommender, load_model_from_checkpoint


@dataclass
class ServingBundle:
    model: TwoTowerRecommender
    checkpoint: dict
    device: torch.device
    users: pd.DataFrame
    markets: pd.DataFrame
    market_embeddings: torch.Tensor


def load_serving_bundle(model_path: str | Path | None = None, device: torch.device | None = None) -> ServingBundle:
    """Load the trained model and all serving-time artifacts."""
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(model_path or MODELS_DIR / "best_model.pt")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=target_device)

    users = pd.read_csv(PROCESSED_DATA_DIR / "users_processed.csv").set_index("user_id")
    markets = pd.read_csv(PROCESSED_DATA_DIR / "markets_processed.csv")
    market_ids = torch.tensor(markets["market_id"].astype(int).to_numpy(), dtype=torch.long, device=target_device)
    market_features = torch.tensor(
        markets[MARKET_FEATURE_COLUMNS].astype(float).to_numpy(),
        dtype=torch.float32,
        device=target_device,
    )
    with torch.no_grad():
        market_embeddings = model.get_market_embedding(market_ids, market_features)

    return ServingBundle(
        model=model,
        checkpoint=checkpoint,
        device=target_device,
        users=users,
        markets=markets,
        market_embeddings=market_embeddings,
    )


def recommend_for_user(
    bundle: ServingBundle,
    user_id: int,
    top_k: int = 10,
    sport_filter: str | None = None,
    exclude_live: bool = False,
) -> list[dict]:
    """Generate top-k recommendations for a user from the precomputed market index."""
    if user_id not in bundle.users.index:
        raise KeyError(f"Unknown user_id: {user_id}")

    # Responsible Gambling gate — no recommendations for flagged users
    user_row_check = bundle.users.loc[user_id]
    if isinstance(user_row_check, pd.DataFrame):
        user_row_check = user_row_check.iloc[0]
    if bool(user_row_check.get("responsible_gambling_flag", False)):
        raise ValueError(
            f"User {user_id} has an active responsible gambling flag. "
            "Recommendations are restricted for this account."
        )

    candidate_markets = bundle.markets.copy()
    candidate_indices = candidate_markets.index

    if sport_filter:
        candidate_markets = candidate_markets[candidate_markets["sport"] == sport_filter]
        candidate_indices = candidate_markets.index
    if exclude_live:
        candidate_markets = candidate_markets[candidate_markets["is_live"] == False]  # noqa: E712
        candidate_indices = candidate_markets.index

    if candidate_markets.empty:
        return []

    user_row = bundle.users.loc[user_id]
    if isinstance(user_row, pd.DataFrame):
        user_row = user_row.iloc[0]

    user_id_tensor = torch.tensor([int(user_id)], dtype=torch.long, device=bundle.device)
    user_feature_tensor = torch.tensor(
        [[float(user_row[column]) for column in USER_FEATURE_COLUMNS]],
        dtype=torch.float32,
        device=bundle.device,
    )

    with torch.no_grad():
        user_embedding = bundle.model.get_user_embedding(user_id_tensor, user_feature_tensor)
        candidate_embeddings = bundle.market_embeddings[candidate_indices]
        scores = torch.matmul(user_embedding, candidate_embeddings.T).squeeze(0)
        top_limit = min(top_k, len(candidate_markets))
        top_scores, top_positions = torch.topk(scores, k=top_limit)

    ranked_rows = candidate_markets.iloc[top_positions.cpu().numpy()].reset_index(drop=True)
    recommendations = []
    for row, score in zip(ranked_rows.to_dict(orient="records"), top_scores.cpu().tolist()):
        recommendations.append(
            {
                "market_id": int(row["market_id"]),
                "sport": row["sport"],
                "market_type": row["market_type"],
                "score": round(float(score), 4),
                "is_live": bool(row["is_live"]),
                "event_prominence": row["event_prominence"],
                "popularity_score": float(row["popularity_score"]),
            }
        )
    return recommendations
