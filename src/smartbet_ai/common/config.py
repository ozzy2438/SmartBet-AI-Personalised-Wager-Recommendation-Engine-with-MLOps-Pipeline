"""Configuration loading with runtime-safe path handling."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from smartbet_ai.common.paths import DATA_SCHEMA_PATH, MLFLOW_DB_PATH, MODEL_CONFIG_PATH


def _set_nested_value(config: dict[str, Any], path: str, value: Any) -> None:
    node = config
    parts = path.split(".")
    for key in parts[:-1]:
        node = node[key]
    node[parts[-1]] = value


def _apply_runtime_overrides(config: dict[str, Any]) -> dict[str, Any]:
    overrides = {
        "SMARTBET_TRAINING_EPOCHS": ("training.epochs", int),
        "SMARTBET_BATCH_SIZE": ("training.batch_size", int),
        "SMARTBET_N_NEGATIVES": ("training.n_negatives", int),
        "SMARTBET_N_USERS": ("data.n_users", int),
        "SMARTBET_N_MARKETS": ("data.n_markets", int),
        "SMARTBET_N_INTERACTIONS": ("data.n_interactions", int),
    }

    effective = copy.deepcopy(config)
    for env_key, (path, cast) in overrides.items():
        raw_value = os.getenv(env_key)
        if raw_value:
            _set_nested_value(effective, path, cast(raw_value))

    tracking_uri = effective["mlops"].get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    if tracking_uri == "sqlite:///mlflow.db":
        effective["mlops"]["mlflow_tracking_uri"] = f"sqlite:///{MLFLOW_DB_PATH}"

    return effective


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _apply_runtime_overrides(config)


def load_data_schema(path: Path = DATA_SCHEMA_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
