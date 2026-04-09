"""
Production API for serving real-time recommendations.

Endpoints:
- GET  /health
- GET  /model-info
- POST /recommend
- POST /ab-test
"""

from __future__ import annotations

import hashlib
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
    version="2.1.0",
)

serving_bundle: ServingBundle | None = None
load_error: str | None = None


# ── Request / Response models ─────────────────────────────────────────────────

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


class ABTestRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_k: int = Field(default=10, ge=1, le=50)
    sport_filter: Optional[str] = None
    exclude_live: bool = False


class ABTestResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    user_id: int
    assigned_variant: str          # "control" or "treatment"
    recommendations: list[dict[str, Any]]
    model_version: str
    experiment: str


# ── Startup ───────────────────────────────────────────────────────────────────

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


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
    summary: dict = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

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
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
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


@app.post("/ab-test", response_model=ABTestResponse)
async def ab_test(request: ABTestRequest) -> ABTestResponse:
    """
    A/B Test endpoint.

    Deterministic variant assignment: same user_id always gets the same variant.
    - control   (50 %) → standard top-k recommendations (default ranking)
    - treatment (50 %) → popularity-boosted ranking (score * 0.7 + popularity * 0.3)

    In a real production system you would swap in a different model checkpoint
    for the treatment arm. Here we simulate the difference via a re-ranking
    strategy so the experiment can run without a second model file.
    """
    if serving_bundle is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    # Deterministic 50/50 split — same user always lands in the same bucket
    digest = int(hashlib.md5(str(request.user_id).encode()).hexdigest(), 16)
    variant = "control" if digest % 2 == 0 else "treatment"

    try:
        recommendations = recommend_for_user(
            bundle=serving_bundle,
            user_id=request.user_id,
            top_k=request.top_k * 2 if variant == "treatment" else request.top_k,
            sport_filter=request.sport_filter,
            exclude_live=request.exclude_live,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations available for the requested filters")

    # Treatment arm: re-rank by blending model score with popularity
    if variant == "treatment":
        max_score = max(abs(r["score"]) for r in recommendations) or 1.0
        for r in recommendations:
            normalised_score = (r["score"] + max_score) / (2 * max_score)
            r["blended_score"] = round(normalised_score * 0.7 + r["popularity_score"] * 0.3, 4)
        recommendations = sorted(recommendations, key=lambda r: r["blended_score"], reverse=True)[: request.top_k]

    checkpoint_epoch = serving_bundle.checkpoint.get("epoch", "unknown")
    return ABTestResponse(
        user_id=request.user_id,
        assigned_variant=variant,
        recommendations=recommendations,
        model_version=f"checkpoint_epoch_{checkpoint_epoch}",
        experiment="popularity_boost_v1",
    )
