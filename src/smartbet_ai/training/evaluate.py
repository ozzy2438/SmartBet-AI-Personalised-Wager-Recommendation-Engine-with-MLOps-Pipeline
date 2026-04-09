"""Evaluation pipeline for the two-tower recommender."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.constants import MARKET_FEATURE_COLUMNS, USER_FEATURE_COLUMNS
from smartbet_ai.common.paths import MODELS_DIR, PROCESSED_DATA_DIR
from smartbet_ai.modeling.model import load_model_from_checkpoint


def compute_ndcg(relevances: list[float], k: int) -> float:
    relevances_array = np.asarray(relevances[:k], dtype=float)
    if relevances_array.size == 0:
        return 0.0
    positions = np.arange(1, len(relevances_array) + 1)
    dcg = np.sum(relevances_array / np.log2(positions + 1))
    ideal = np.sort(relevances_array)[::-1]
    idcg = np.sum(ideal / np.log2(positions + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_precision(relevances: list[float], k: int) -> float:
    sliced = relevances[:k]
    return float(np.mean(sliced)) if sliced else 0.0


def compute_recall(relevances: list[float], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return float(np.sum(relevances[:k]) / total_relevant)


def _get_mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def _sanitize_mlflow_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    return {name.replace("@", "_at_"): value for name, value in metrics.items()}


def evaluate(model_path: str | None = None, k_values: list[int] | None = None) -> dict[str, float]:
    """Evaluate the saved model on ranking metrics."""
    config = load_model_config()
    k_values = k_values or [5, 10, 20]
    checkpoint_path = MODELS_DIR / "best_model.pt" if model_path is None else model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    saved_config = checkpoint["config"]
    data_config = saved_config["data"]

    users = pd.read_csv(PROCESSED_DATA_DIR / "users_processed.csv").set_index("user_id")
    markets = pd.read_csv(PROCESSED_DATA_DIR / "markets_processed.csv")
    interactions = pd.read_csv(PROCESSED_DATA_DIR / "interactions_processed.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values("timestamp")

    n_rows = len(interactions)
    val_end = int(n_rows * (data_config["train_ratio"] + data_config["val_ratio"]))
    test_df = interactions.iloc[val_end:]
    user_test_positives = test_df.groupby("user_id")["market_id"].apply(lambda values: set(values.astype(int))).to_dict()

    all_market_ids = torch.tensor(markets["market_id"].astype(int).to_numpy(), dtype=torch.long, device=device)
    all_market_features = torch.tensor(
        markets[MARKET_FEATURE_COLUMNS].astype(float).to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    print("Precomputing market embeddings...")
    with torch.no_grad():
        all_market_embeddings = model.get_market_embedding(all_market_ids, all_market_features)

    metric_buckets: dict[str, list[float]] = defaultdict(list)
    print("Evaluating per-user recommendations...")

    with torch.no_grad():
        for user_id, positive_markets in user_test_positives.items():
            if not positive_markets or user_id not in users.index:
                continue

            user_row = users.loc[user_id]
            if isinstance(user_row, pd.DataFrame):
                user_row = user_row.iloc[0]

            user_id_tensor = torch.tensor([int(user_id)], dtype=torch.long, device=device)
            user_feature_tensor = torch.tensor(
                [[float(user_row[column]) for column in USER_FEATURE_COLUMNS]],
                dtype=torch.float32,
                device=device,
            )
            user_embedding = model.get_user_embedding(user_id_tensor, user_feature_tensor)
            scores = torch.matmul(user_embedding, all_market_embeddings.T).squeeze(0)
            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
            ranked_market_ids = markets["market_id"].to_numpy()[ranked_indices]
            relevances = [1.0 if int(market_id) in positive_markets else 0.0 for market_id in ranked_market_ids]
            total_relevant = len(positive_markets)

            for k in k_values:
                metric_buckets[f"ndcg@{k}"].append(compute_ndcg(relevances, k))
                metric_buckets[f"precision@{k}"].append(compute_precision(relevances, k))
                metric_buckets[f"recall@{k}"].append(compute_recall(relevances, k, total_relevant))
                metric_buckets[f"hit_rate@{k}"].append(1.0 if sum(relevances[:k]) > 0 else 0.0)

    results = {metric: float(np.mean(values)) if values else 0.0 for metric, values in metric_buckets.items()}

    print("\nEvaluation results")
    print("=" * 50)
    for name, value in sorted(results.items()):
        print(f"{name}: {value:.4f}")
    print("=" * 50)

    with (MODELS_DIR / "evaluation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    mlflow = _get_mlflow()
    if mlflow is not None:
        mlflow.set_tracking_uri(config["mlops"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["mlops"]["experiment_name"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(_sanitize_mlflow_metric_names(results))

    return results


if __name__ == "__main__":
    evaluate()
