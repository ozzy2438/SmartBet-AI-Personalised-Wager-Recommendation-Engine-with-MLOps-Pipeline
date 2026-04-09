"""Training loop for the two-tower recommender with optional MLflow tracking."""

from __future__ import annotations

import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.paths import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from smartbet_ai.features.engineering import engineer_features
from smartbet_ai.modeling.dataset import BettingRecommendationDataset
from smartbet_ai.modeling.model import build_model_from_config


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_mlflow():
    try:
        import mlflow
        import mlflow.pytorch

        return mlflow
    except ImportError:
        return None


def train() -> dict:
    """Run model training and persist the best checkpoint and summary."""
    _seed_everything()
    ensure_project_dirs()
    config = load_model_config()
    training_config = config["training"]
    data_config = config["data"]
    mlops_config = config["mlops"]

    if not (PROCESSED_DATA_DIR / "users_processed.csv").exists():
        print("Processed artifacts not found. Running feature engineering...")
        engineer_features()

    users = pd.read_csv(PROCESSED_DATA_DIR / "users_processed.csv")
    markets = pd.read_csv(PROCESSED_DATA_DIR / "markets_processed.csv")
    interactions = pd.read_csv(PROCESSED_DATA_DIR / "interactions_processed.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    n_rows = len(interactions)
    train_end = int(n_rows * data_config["train_ratio"])
    val_end = int(n_rows * (data_config["train_ratio"] + data_config["val_ratio"]))

    train_df = interactions.iloc[:train_end]
    val_df = interactions.iloc[train_end:val_end]
    test_df = interactions.iloc[val_end:]
    print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    train_dataset = BettingRecommendationDataset(
        train_df,
        users,
        markets,
        n_negatives=training_config["n_negatives"],
        split="train",
    )
    val_dataset = BettingRecommendationDataset(
        val_df,
        users,
        markets,
        n_negatives=training_config["n_negatives"],
        split="val",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(training_config["epochs"], 1),
        eta_min=1e-6,
    )

    mlflow = _get_mlflow()
    active_run = None
    if mlflow is not None:
        mlflow.set_tracking_uri(mlops_config["mlflow_tracking_uri"])
        mlflow.set_experiment(mlops_config["experiment_name"])
        active_run = mlflow.start_run(run_name=f"two_tower_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params(
            {
                "embedding_dim": config["model"]["embedding_dim"],
                "final_embedding_dim": config["model"]["final_embedding_dim"],
                "user_hidden_dims": str(config["model"]["user_tower"]["hidden_dims"]),
                "market_hidden_dims": str(config["model"]["market_tower"]["hidden_dims"]),
                "temperature": config["model"]["temperature"],
                "learning_rate": training_config["learning_rate"],
                "batch_size": training_config["batch_size"],
                "epochs": training_config["epochs"],
                "weight_decay": training_config["weight_decay"],
                "n_users": data_config["n_users"],
                "n_markets": data_config["n_markets"],
                "n_interactions": data_config["n_interactions"],
                "n_negatives": training_config["n_negatives"],
                "device": str(device),
            }
        )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_config["epochs"]):
        model.train()
        train_losses: list[float] = []

        for user_ids, user_features, market_ids, market_features, labels in train_loader:
            user_ids = user_ids.to(device)
            user_features = user_features.to(device)
            market_ids = market_ids.to(device)
            market_features = market_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(user_ids, user_features, market_ids, market_features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for user_ids, user_features, market_ids, market_features, labels in val_loader:
                user_ids = user_ids.to(device)
                user_features = user_features.to(device)
                market_ids = market_ids.to(device)
                market_features = market_features.to(device)
                labels = labels.to(device)

                logits, _, _ = model(user_ids, user_features, market_ids, market_features)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

        average_train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        average_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        current_temperature = float(torch.clamp(model.temperature.detach(), min=1e-3).item())
        print(
            f"Epoch {epoch + 1}/{training_config['epochs']} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {average_val_loss:.4f} | "
            f"LR: {current_lr:.6f} | Temp: {current_temperature:.4f}"
        )

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "train_loss": average_train_loss,
                    "val_loss": average_val_loss,
                    "learning_rate": current_lr,
                    "temperature": current_temperature,
                },
                step=epoch,
            )

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }
            torch.save(checkpoint, MODELS_DIR / "best_model.pt")
            print(f"Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= training_config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    summary = {
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch + 1,
        "final_temperature": float(torch.clamp(model.temperature.detach(), min=1e-3).item()),
        "training_completed": datetime.now().isoformat(),
        "device": str(device),
    }
    with (MODELS_DIR / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if mlflow is not None:
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.end_run()
    elif active_run is not None:
        active_run.end()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return summary


if __name__ == "__main__":
    train()
