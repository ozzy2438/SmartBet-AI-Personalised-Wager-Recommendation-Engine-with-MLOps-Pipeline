"""Register the latest trained model in MLflow and a local registry file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.paths import MODELS_DIR


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def register_model(model_path: str | Path | None = None, stage: str = "staging") -> dict[str, Any]:
    """Register a model in the local registry and attempt MLflow registration when available."""
    config = load_model_config()
    registry_path = MODELS_DIR / "model_registry.json"
    checkpoint_path = Path(model_path or MODELS_DIR / "best_model.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    registry = _load_json_if_exists(registry_path)
    versions = registry.get("versions", [])
    next_version = (max((entry["version"] for entry in versions), default=0) + 1) if versions else 1
    metrics = _load_json_if_exists(MODELS_DIR / "evaluation_results.json")
    training_summary = _load_json_if_exists(MODELS_DIR / "training_summary.json")

    record = {
        "version": next_version,
        "stage": stage,
        "path": str(checkpoint_path),
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "training_summary": training_summary,
    }
    versions.append(record)
    registry["versions"] = versions

    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)

    mlflow_result: dict[str, Any] | None = None
    try:
        import mlflow
        import mlflow.pytorch
        import torch

        from smartbet_ai.modeling.model import load_model_from_checkpoint

        mlflow.set_tracking_uri(config["mlops"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["mlops"]["experiment_name"])
        model, _checkpoint = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"))
        with mlflow.start_run(run_name=f"register_model_v{next_version}"):
            mlflow.pytorch.log_model(model, artifact_path="registered_model")
            mlflow.log_params({"registry_version": next_version, "stage": stage})
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name.replace("@", "_at_"), float(metric_value))
        mlflow_result = {"status": "logged_to_mlflow"}
    except Exception as exc:
        mlflow_result = {"status": "mlflow_skipped", "reason": str(exc)}

    response = {"local_registry": record, "mlflow": mlflow_result}
    print(json.dumps(response, indent=2))
    return response


if __name__ == "__main__":
    register_model()
