"""Data drift detection using PSI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def compute_psi(reference_series: pd.Series, current_series: pd.Series, bins: int = 10) -> float:
    """Compute population stability index for a single feature."""
    reference = reference_series.dropna().to_numpy()
    current = current_series.dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    _, bin_edges = np.histogram(reference, bins=bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    epsilon = 1e-6
    reference_pct = np.clip(reference_counts / len(reference), epsilon, 1.0)
    current_pct = np.clip(current_counts / len(current), epsilon, 1.0)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def check_drift(
    reference_path: str | None = None,
    current_path: str | None = None,
    threshold: float | None = None,
) -> tuple[dict[str, dict[str, float | str]], bool]:
    """Compare a processed reference window to a current raw window."""
    config = load_model_config()
    psi_threshold = threshold if threshold is not None else config["mlops"]["drift_threshold"]

    reference_df = pd.read_csv(reference_path or (PROCESSED_DATA_DIR / "interactions_processed.csv"))
    current_df = pd.read_csv(current_path or (RAW_DATA_DIR / "interactions.csv"))

    numeric_features = ["stake", "odds"]
    categorical_features = ["sport", "market_type", "outcome"]
    drift_results: dict[str, dict[str, float | str]] = {}
    overall_drift = False

    print("\nDrift detection report")
    print("=" * 50)

    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi = compute_psi(reference_df[feature], current_df[feature])
        status = "GREEN" if psi < psi_threshold else "YELLOW" if psi < 0.25 else "RED"
        drift_results[feature] = {"psi": float(psi), "status": status}
        print(f"{feature}: PSI={psi:.4f} [{status}]")
        if psi >= psi_threshold:
            overall_drift = True

    for feature in categorical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref_freq = reference_df[feature].value_counts(normalize=True)
        cur_freq = current_df[feature].value_counts(normalize=True)
        categories = sorted(set(ref_freq.index) | set(cur_freq.index))
        ref_vals = np.array([ref_freq.get(category, 1e-6) for category in categories])
        cur_vals = np.array([cur_freq.get(category, 1e-6) for category in categories])
        psi = float(np.sum((cur_vals - ref_vals) * np.log(cur_vals / ref_vals)))
        status = "GREEN" if psi < psi_threshold else "YELLOW" if psi < 0.25 else "RED"
        drift_results[feature] = {"psi": psi, "status": status}
        print(f"{feature}: PSI={psi:.4f} [{status}]")
        if psi >= psi_threshold:
            overall_drift = True

    with (MODELS_DIR / "drift_report.json").open("w", encoding="utf-8") as handle:
        json.dump(drift_results, handle, indent=2)

    if overall_drift:
        print("Drift detected. Consider retraining.")
    else:
        print("No significant drift detected.")

    return drift_results, overall_drift


if __name__ == "__main__":
    check_drift()
