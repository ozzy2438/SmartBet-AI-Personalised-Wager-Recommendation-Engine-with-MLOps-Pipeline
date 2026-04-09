"""Lightweight Streamlit dashboard for model monitoring and local recommendations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from smartbet_ai.common.paths import MODELS_DIR
from smartbet_ai.modeling.inference import load_serving_bundle, recommend_for_user


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


@st.cache_resource
def _get_bundle():
    try:
        return load_serving_bundle()
    except Exception:
        return None


st.set_page_config(page_title="SmartBet AI Dashboard", layout="wide")
st.title("SmartBet AI Dashboard")
st.caption("Stage-1 operational dashboard over local training artifacts")

training_summary = _load_json_if_exists(MODELS_DIR / "training_summary.json")
evaluation_results = _load_json_if_exists(MODELS_DIR / "evaluation_results.json")
drift_report = _load_json_if_exists(MODELS_DIR / "drift_report.json")
registry = _load_json_if_exists(MODELS_DIR / "model_registry.json")

col1, col2, col3 = st.columns(3)
col1.metric("Best Val Loss", training_summary.get("best_val_loss", "N/A"))
col2.metric("Epochs Trained", training_summary.get("epochs_trained", "N/A"))
col3.metric("NDCG@10", evaluation_results.get("ndcg@10", "N/A"))

with st.expander("Training Summary", expanded=True):
    st.json(training_summary or {"message": "No training summary found yet."})

with st.expander("Evaluation Metrics", expanded=True):
    st.json(evaluation_results or {"message": "No evaluation results found yet."})

with st.expander("Drift Report"):
    st.json(drift_report or {"message": "No drift report found yet."})

with st.expander("Model Registry"):
    st.json(registry or {"message": "No registered models found yet."})

bundle = _get_bundle()
st.subheader("Local Recommendation Preview")
if bundle is None:
    st.info("A trained model and processed artifacts are required before recommendations can be previewed here.")
else:
    available_users = bundle.users.index.tolist()
    available_sports = sorted(bundle.markets["sport"].dropna().unique().tolist())

    selected_user = st.selectbox("User ID", options=available_users[:500] if len(available_users) > 500 else available_users)
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    sport_filter = st.selectbox("Sport Filter", options=["All"] + available_sports)
    exclude_live = st.checkbox("Exclude live markets", value=False)

    if st.button("Generate Recommendations"):
        recommendations = recommend_for_user(
            bundle=bundle,
            user_id=int(selected_user),
            top_k=top_k,
            sport_filter=None if sport_filter == "All" else sport_filter,
            exclude_live=exclude_live,
        )
        if recommendations:
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.warning("No recommendations found for the current filter set.")
