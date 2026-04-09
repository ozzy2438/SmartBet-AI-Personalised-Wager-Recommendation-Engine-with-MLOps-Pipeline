"""
Production API for serving real-time recommendations.

Endpoints:
- POST /recommend
- GET /health
- GET /model-info
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from smartbet_ai.common.paths import MODELS_DIR
from smartbet_ai.modeling.inference import ServingBundle, load_serving_bundle, recommend_for_user

app = FastAPI(
    title="SmartBet AI - Wager Recommendation Engine",
    description="Production API for personalised sports betting market recommendations",
    version="1.0.0",
)

serving_bundle: ServingBundle | None = None
load_error: str | None = None


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_k: int = Field(default=10, ge=1, le=50)
    sport_filter: Optional[str] = None
    exclude_live: bool = False


class RecommendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    user_id: int
    recommendations: list[dict[str, Any]]
    model_version: str


@app.on_event("startup")
async def load_model_and_data() -> None:
    """Load the trained model and market embeddings during app startup."""
    global serving_bundle, load_error

    try:
        serving_bundle = load_serving_bundle()
        load_error = None
        print("Model loaded and market embeddings precomputed")
        print(f"{len(serving_bundle.markets)} markets indexed and ready for serving")
    except Exception as exc:
        serving_bundle = None
        load_error = str(exc)
        print(f"Startup warning: {load_error}")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy" if serving_bundle is not None else "degraded",
        "model_loaded": serving_bundle is not None,
        "load_error": load_error,
    }


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    if serving_bundle is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    summary_path = MODELS_DIR / "training_summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

    config = serving_bundle.checkpoint["config"]
    return {
        "model_type": "Two-Tower Deep Learning Recommender",
        "embedding_dim": config["model"]["final_embedding_dim"],
        "temperature": float(serving_bundle.model.temperature.detach().item()),
        "num_market_embeddings": int(len(serving_bundle.markets)),
        "num_users_indexed": int(len(serving_bundle.users)),
        "training_summary": summary,
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    if serving_bundle is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    try:
        recommendations = recommend_for_user(
            bundle=serving_bundle,
            user_id=request.user_id,
            top_k=request.top_k,
            sport_filter=request.sport_filter,
            exclude_live=request.exclude_live,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations available for the requested filters")

    checkpoint_epoch = serving_bundle.checkpoint.get("epoch", "unknown")
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version=f"checkpoint_epoch_{checkpoint_epoch}",
    )
