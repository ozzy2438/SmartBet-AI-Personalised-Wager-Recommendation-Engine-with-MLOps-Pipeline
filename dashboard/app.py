"""Professional SmartBet AI Dashboard — Stage-1 MLOps Monitoring & Recommendation Engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbet_ai.common.paths import MODELS_DIR
from smartbet_ai.modeling.inference import load_serving_bundle, recommend_for_user

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartBet AI | MLOps Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font & base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background-color: #0f1117; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d27 0%, #141720 100%);
    border-right: 1px solid #2d3147;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
  }
  .kpi-sub {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
  }
  .kpi-badge-green  { color: #22c55e; font-size:12px; font-weight:600; }
  .kpi-badge-yellow { color: #eab308; font-size:12px; font-weight:600; }
  .kpi-badge-red    { color: #ef4444; font-size:12px; font-weight:600; }

  /* Section headers */
  .section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1e2235;
    margin-bottom: 16px;
  }

  /* Drift pill */
  .drift-pill-green  { background:#14532d; color:#4ade80; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .drift-pill-yellow { background:#713f12; color:#fde047; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .drift-pill-red    { background:#7f1d1d; color:#f87171; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

  /* Tables */
  .stDataFrame { border-radius: 10px; overflow: hidden; }
  [data-testid="stDataFrame"] table { background: #1e2235 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* Selectbox / slider labels */
  label { color: #94a3b8 !important; font-size: 13px !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Divider */
  hr { border-color: #1e2235; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def _get_bundle():
    try:
        return load_serving_bundle()
    except Exception:
        return None


def _gauge(value: float, title: str, threshold: float = 0.6) -> go.Figure:
    color = "#22c55e" if value >= threshold else "#eab308" if value >= threshold * 0.6 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 4),
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        number={"font": {"size": 28, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#475569", "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2235",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 0.6], "color": "#1e2235"},
                {"range": [threshold * 0.6, threshold], "color": "#1e2235"},
                {"range": [threshold, 1], "color": "#1e2235"},
            ],
            "threshold": {"line": {"color": "#6366f1", "width": 2}, "thickness": 0.75, "value": threshold},
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="#1e2235",
        font={"family": "Inter"},
    )
    return fig


def _bar_metrics(eval_results: dict) -> go.Figure:
    ks = [5, 10, 20]
    metrics = ["ndcg", "hit_rate", "precision", "recall"]
    colors = {"ndcg": "#6366f1", "hit_rate": "#22c55e", "precision": "#f59e0b", "recall": "#38bdf8"}
    labels = {"ndcg": "NDCG", "hit_rate": "Hit Rate", "precision": "Precision", "recall": "Recall"}

    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(
            name=labels[m],
            x=[f"@{k}" for k in ks],
            y=[eval_results.get(f"{m}@{k}", 0) for k in ks],
            marker_color=colors[m],
            marker_line_width=0,
            opacity=0.9,
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1e2235",
        plot_bgcolor="#1e2235",
        font={"family": "Inter", "color": "#94a3b8"},
        legend={"bgcolor": "#1e2235", "font": {"size": 11}},
        xaxis={"gridcolor": "#2d3147", "title": ""},
        yaxis={"gridcolor": "#2d3147", "title": "Score", "range": [0, 1]},
        margin=dict(t=10, b=10, l=0, r=0),
        height=280,
    )
    return fig


def _drift_bar(drift_report: dict) -> go.Figure:
    if not drift_report:
        return go.Figure()
    features = list(drift_report.keys())
    psi_values = [drift_report[f].get("psi", 0) for f in features]
    colors = []
    for v in psi_values:
        if v < 0.05:
            colors.append("#22c55e")
        elif v < 0.25:
            colors.append("#eab308")
        else:
            colors.append("#ef4444")

    fig = go.Figure(go.Bar(
        x=features,
        y=psi_values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.4f}" for v in psi_values],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 11},
    ))
    fig.add_hline(y=0.05, line_dash="dot", line_color="#6366f1", annotation_text="Threshold (0.05)",
                  annotation_font_color="#6366f1", annotation_font_size=11)
    fig.update_layout(
        paper_bgcolor="#1e2235",
        plot_bgcolor="#1e2235",
        font={"family": "Inter", "color": "#94a3b8"},
        xaxis={"gridcolor": "#2d3147"},
        yaxis={"gridcolor": "#2d3147", "title": "PSI Score"},
        margin=dict(t=20, b=10, l=0, r=0),
        height=240,
        showlegend=False,
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────
training   = _load(MODELS_DIR / "training_summary.json")
evaluation = _load(MODELS_DIR / "evaluation_results.json")
drift      = _load(MODELS_DIR / "drift_report.json")
registry   = _load(MODELS_DIR / "model_registry.json")
bundle     = _get_bundle()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 SmartBet AI")
    st.markdown("<div style='color:#475569;font-size:12px;margin-bottom:20px'>MLOps Monitoring Platform</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Model Performance", "Data Drift", "Recommendations", "Model Registry"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    model_ok = bundle is not None
    status_color = "#22c55e" if model_ok else "#ef4444"
    status_text  = "Model Loaded" if model_ok else "Model Not Found"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px'>"
        f"<div style='width:8px;height:8px;border-radius:50%;background:{status_color}'></div>"
        f"<span style='color:#94a3b8;font-size:12px'>{status_text}</span></div>",
        unsafe_allow_html=True,
    )

    versions = registry.get("versions", [])
    if versions:
        latest = versions[-1]
        st.markdown(
            f"<div style='margin-top:8px;color:#475569;font-size:11px'>"
            f"Version {latest['version']} · {latest['stage'].upper()}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='position:absolute;bottom:20px;color:#2d3147;font-size:10px'>Stage-1 Build · 2026</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("## SmartBet AI — Wager Recommendation Engine")
    st.markdown("<div style='color:#475569;font-size:14px;margin-bottom:28px'>"
                "Personalised sports betting market recommendations powered by a Two-Tower deep learning model.</div>",
                unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    val_loss  = training.get("best_val_loss", 0)
    epochs    = training.get("epochs_trained", 0)
    ndcg10    = evaluation.get("ndcg@10", 0)
    hit10     = evaluation.get("hit_rate@10", 0)
    temp      = training.get("final_temperature", 0)
    device    = training.get("device", "cpu").upper()
    completed = training.get("training_completed", "—")[:10] if training else "—"

    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Best Val Loss</div>
            <div class="kpi-value">{val_loss:.4f}</div>
            <div class="kpi-sub">{epochs} epochs · {device}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        badge = "kpi-badge-yellow" if ndcg10 < 0.6 else "kpi-badge-green"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">NDCG @ 10</div>
            <div class="kpi-value">{ndcg10:.3f}</div>
            <div class="kpi-sub"><span class="{badge}">Target 0.600</span></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Hit Rate @ 10</div>
            <div class="kpi-value">{hit10:.1%}</div>
            <div class="kpi-sub">Top-10 recommendation accuracy</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        drift_ok = all(v.get("status") == "GREEN" for v in drift.values()) if drift else False
        d_label  = "ALL CLEAR" if drift_ok else "DRIFT DETECTED"
        d_badge  = "kpi-badge-green" if drift_ok else "kpi-badge-red"
        monitored = len(drift) if drift else 0
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Data Drift Status</div>
            <div class="kpi-value"><span class="{d_badge}">{d_label}</span></div>
            <div class="kpi-sub">{monitored} features monitored</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gauges + metrics preview
    col_l, col_r = st.columns([1.2, 2])
    with col_l:
        st.markdown("<div class='section-title'>Key Ranking Metrics</div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(_gauge(ndcg10, "NDCG@10", 0.6), use_container_width=True, config={"displayModeBar": False})
        with g2:
            st.plotly_chart(_gauge(hit10, "Hit Rate@10", 0.5), use_container_width=True, config={"displayModeBar": False})
        g3, g4 = st.columns(2)
        with g3:
            st.plotly_chart(_gauge(evaluation.get("ndcg@20", 0), "NDCG@20", 0.6), use_container_width=True, config={"displayModeBar": False})
        with g4:
            st.plotly_chart(_gauge(evaluation.get("hit_rate@20", 0), "Hit Rate@20", 0.5), use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown("<div class='section-title'>Evaluation Metrics Breakdown</div>", unsafe_allow_html=True)
        if evaluation:
            st.plotly_chart(_bar_metrics(evaluation), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No evaluation results yet.")

        # Drift row
        st.markdown("<div class='section-title'>Feature Drift (PSI)</div>", unsafe_allow_html=True)
        if drift:
            st.plotly_chart(_drift_bar(drift), use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.markdown("<div style='color:#475569;font-size:14px;margin-bottom:24px'>Detailed ranking metrics across cut-off values.</div>",
                unsafe_allow_html=True)

    if not evaluation:
        st.warning("No evaluation results found. Run `python src/evaluate.py` first.")
    else:
        st.markdown("<div class='section-title'>Metric Overview</div>", unsafe_allow_html=True)
        st.plotly_chart(_bar_metrics(evaluation), use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-title'>Full Metrics Table</div>", unsafe_allow_html=True)
        rows = []
        for key, val in sorted(evaluation.items()):
            metric, k = key.split("@")
            rows.append({"Metric": metric.upper(), "K": int(k), "Score": round(val, 5),
                         "Target": "0.600" if metric == "ndcg" else "—",
                         "Status": "✅" if (metric == "ndcg" and val >= 0.6) else ("⚠️" if metric == "ndcg" else "—")})
        st.dataframe(
            pd.DataFrame(rows).sort_values(["Metric", "K"]),
            use_container_width=True,
            hide_index=True,
        )

    if training:
        st.markdown("---")
        st.markdown("<div class='section-title'>Training Summary</div>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Best Val Loss", f"{training.get('best_val_loss', 0):.4f}")
        t2.metric("Epochs Trained", training.get("epochs_trained", "—"))
        t3.metric("Temperature", f"{training.get('final_temperature', 0):.4f}")
        t4.metric("Device", training.get("device", "—").upper())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Drift":
    st.markdown("## Data Drift Monitor")
    st.markdown("<div style='color:#475569;font-size:14px;margin-bottom:24px'>"
                "Population Stability Index (PSI) per feature. PSI &lt; 0.05 = stable. 0.05–0.25 = warning. &gt; 0.25 = critical.</div>",
                unsafe_allow_html=True)

    if not drift:
        st.warning("No drift report found. Run `python src/drift_check.py` first.")
    else:
        all_green = all(v.get("status") == "GREEN" for v in drift.values())
        if all_green:
            st.success("✅  All features are stable. No retraining required.")
        else:
            st.error("🚨  Drift detected on one or more features. Consider retraining.")

        st.markdown("<div class='section-title'>PSI by Feature</div>", unsafe_allow_html=True)
        st.plotly_chart(_drift_bar(drift), use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-title'>Feature Detail</div>", unsafe_allow_html=True)
        rows = []
        pill_map = {"GREEN": "drift-pill-green", "YELLOW": "drift-pill-yellow", "RED": "drift-pill-red"}
        for feature, info in drift.items():
            status = info.get("status", "GREEN")
            psi    = info.get("psi", 0)
            pill   = pill_map.get(status, "drift-pill-green")
            rows.append({
                "Feature": feature,
                "PSI": round(psi, 6),
                "Status": status,
                "Action": "No action" if status == "GREEN" else ("Monitor closely" if status == "YELLOW" else "Retrain immediately"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("## Live Recommendation Preview")
    st.markdown("<div style='color:#475569;font-size:14px;margin-bottom:24px'>"
                "Generate personalised market recommendations from the trained model.</div>",
                unsafe_allow_html=True)

    if bundle is None:
        st.error("Model not loaded. Run the training pipeline first.")
    else:
        available_users  = bundle.users.index.tolist()
        available_sports = sorted(bundle.markets["sport"].dropna().unique().tolist())

        col_a, col_b, col_c, col_d = st.columns([1.5, 1, 1.5, 1])
        with col_a:
            selected_user = st.selectbox("User ID", options=available_users[:500] if len(available_users) > 500 else available_users)
        with col_b:
            top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
        with col_c:
            sport_filter = st.selectbox("Sport Filter", options=["All Sports"] + available_sports)
        with col_d:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            exclude_live = st.checkbox("Exclude Live Markets")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Generate Recommendations"):
            with st.spinner("Computing recommendations..."):
                recs = recommend_for_user(
                    bundle=bundle,
                    user_id=int(selected_user),
                    top_k=top_k,
                    sport_filter=None if sport_filter == "All Sports" else sport_filter,
                    exclude_live=exclude_live,
                )

            if recs:
                st.markdown(f"<div style='color:#64748b;font-size:12px;margin-bottom:12px'>"
                            f"{len(recs)} recommendations for User {selected_user}</div>",
                            unsafe_allow_html=True)

                df = pd.DataFrame(recs)
                df["rank"] = range(1, len(df) + 1)
                df["score"] = df["score"].round(4)
                df["is_live"] = df["is_live"].map({True: "🔴 LIVE", False: "—"})
                df["popularity_score"] = df["popularity_score"].round(4)
                df["event_prominence"] = df["event_prominence"].str.upper()
                df = df[["rank", "market_id", "sport", "market_type", "score", "event_prominence", "popularity_score", "is_live"]]
                df.columns = ["#", "Market ID", "Sport", "Market Type", "Score", "Prominence", "Popularity", "Live"]

                st.dataframe(df, use_container_width=True, hide_index=True)

                # Mini bar chart of scores
                fig = go.Figure(go.Bar(
                    x=[f"#{i}" for i in df["#"]],
                    y=df["Score"],
                    marker_color="#6366f1",
                    marker_line_width=0,
                ))
                fig.update_layout(
                    paper_bgcolor="#1e2235", plot_bgcolor="#1e2235",
                    font={"family": "Inter", "color": "#94a3b8"},
                    xaxis={"gridcolor": "#2d3147", "title": "Recommendation Rank"},
                    yaxis={"gridcolor": "#2d3147", "title": "Model Score"},
                    height=200, margin=dict(t=10, b=10, l=0, r=0),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("No recommendations returned for the selected filters.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Registry":
    st.markdown("## Model Registry")
    st.markdown("<div style='color:#475569;font-size:14px;margin-bottom:24px'>"
                "Version history and promotion pipeline for deployed models.</div>",
                unsafe_allow_html=True)

    versions = registry.get("versions", [])
    if not versions:
        st.warning("No registered models found.")
    else:
        stage_colors = {"staging": "#f59e0b", "production": "#22c55e", "archived": "#6b7280"}

        for v in reversed(versions):
            stage = v.get("stage", "staging")
            color = stage_colors.get(stage, "#6366f1")
            metrics = v.get("metrics", {})
            summary = v.get("training_summary", {})

            st.markdown(f"""
            <div class="kpi-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                    <div style="font-size:16px;font-weight:700;color:#f1f5f9">Version {v.get('version')}</div>
                    <div style="background:{color}22;color:{color};padding:4px 14px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase">{stage}</div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:12px">
                    <div><div style="color:#475569;font-size:11px;margin-bottom:4px">NDCG@10</div>
                         <div style="color:#f1f5f9;font-weight:600">{metrics.get('ndcg@10', 0):.4f}</div></div>
                    <div><div style="color:#475569;font-size:11px;margin-bottom:4px">Hit Rate@10</div>
                         <div style="color:#f1f5f9;font-weight:600">{metrics.get('hit_rate@10', 0):.4f}</div></div>
                    <div><div style="color:#475569;font-size:11px;margin-bottom:4px">Val Loss</div>
                         <div style="color:#f1f5f9;font-weight:600">{summary.get('best_val_loss', 0):.4f}</div></div>
                    <div><div style="color:#475569;font-size:11px;margin-bottom:4px">Epochs</div>
                         <div style="color:#f1f5f9;font-weight:600">{summary.get('epochs_trained', '—')}</div></div>
                </div>
                <div style="color:#475569;font-size:11px">Registered: {v.get('registered_at','—')[:19].replace('T',' ')}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(versions) > 1:
            st.markdown("<div class='section-title'>Version Comparison</div>", unsafe_allow_html=True)
            metrics_to_compare = ["ndcg@5", "ndcg@10", "ndcg@20", "hit_rate@10"]
            fig = go.Figure()
            for v in versions:
                m = v.get("metrics", {})
                fig.add_trace(go.Bar(
                    name=f"v{v['version']} ({v.get('stage','?')})",
                    x=[k.upper().replace("@", " @") for k in metrics_to_compare],
                    y=[m.get(k, 0) for k in metrics_to_compare],
                    marker_line_width=0,
                ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="#1e2235", plot_bgcolor="#1e2235",
                font={"family": "Inter", "color": "#94a3b8"},
                xaxis={"gridcolor": "#2d3147"},
                yaxis={"gridcolor": "#2d3147", "range": [0, 0.5]},
                legend={"bgcolor": "#1e2235"},
                height=280, margin=dict(t=10, b=10, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
