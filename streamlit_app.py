# streamlit_app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from orchestrator.langgraph_pipeline import build_graph
from config import validate_api_key

SNAPSHOT_FILE = ".pipeline_snapshot.json"
_SKIP_STATE_KEYS = {"raw_data", "processed_data", "_progress_callback"}


def _save_snapshot():
    """Write pipeline progress to disk so a page reload can restore it."""
    safe_state = {}
    for k, v in st.session_state.latest_state.items():
        if k in _SKIP_STATE_KEYS:
            continue
        try:
            json.dumps(v, default=str)
            safe_state[k] = v
        except Exception:
            pass
    snap = {
        "log_lines":   st.session_state.log_lines,
        "step_status": st.session_state.step_status,
        "step_times":  st.session_state.step_times,
        "completed":   st.session_state.completed,
        "tuning_live": st.session_state.tuning_live,
        "latest_state": safe_state,
        "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(SNAPSHOT_FILE, "w") as f:
            json.dump(snap, f, default=str)
    except Exception:
        pass


def _restore_snapshot() -> str:
    """Load snapshot. Returns saved_at string, or '' if nothing to restore."""
    if not os.path.exists(SNAPSHOT_FILE):
        return ""
    try:
        with open(SNAPSHOT_FILE) as f:
            snap = json.load(f)
        st.session_state.log_lines    = snap.get("log_lines", [])
        st.session_state.step_status  = snap.get("step_status", {})
        st.session_state.step_times   = snap.get("step_times", {})
        st.session_state.completed    = snap.get("completed", False)
        st.session_state.tuning_live  = snap.get("tuning_live", {})
        st.session_state.latest_state = snap.get("latest_state", {})
        st.session_state.running      = False   # can't resume after reload
        return snap.get("saved_at", "restored")
    except Exception:
        return ""

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic ML System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* Dark background */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12151d;
        border-right: 1px solid #1e2535;
    }

    /* Header */
    .dashboard-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #1e2535;
        margin-bottom: 1.5rem;
    }
    .dashboard-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .dashboard-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }

    /* Metric cards */
    .metric-card {
        background: #12151d;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #3b4a6b; }
    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        font-family: 'Syne', sans-serif;
        color: #f1f5f9;
        line-height: 1;
    }
    .metric-value.good  { color: #34d399; }
    .metric-value.warn  { color: #fbbf24; }
    .metric-value.info  { color: #60a5fa; }

    /* Pipeline step tracker */
    .pipeline-track {
        background: #12151d;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .step-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid #1a1f2e;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .step-row:last-child { border-bottom: none; }
    .step-check {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        flex-shrink: 0;
        border: 1px solid #253049;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.62rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .check-done {
        background: #123126;
        border-color: #34d399;
        color: #34d399;
        box-shadow: 0 0 10px #34d39955;
    }
    .check-running {
        background: #35270f;
        border-color: #fbbf24;
        color: #fbbf24;
        box-shadow: 0 0 12px #fbbf2466;
        animation: pulse 1s infinite;
    }
    .check-pending {
        background: #151b28;
        border-color: #253049;
        color: #40506f;
    }
    .step-name { color: #94a3b8; }
    .step-name.done    { color: #e2e8f0; }
    .step-name.running { color: #fbbf24; font-weight: 600; }
    .step-time { margin-left: auto; color: #3b4a6b; font-size: 0.72rem; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    /* Insight cards */
    .insight-card {
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.5;
    }
    .insight-warn  { background: #2d1f0a; border-left: 3px solid #fbbf24; color: #fde68a; }
    .insight-good  { background: #0a2d1f; border-left: 3px solid #34d399; color: #a7f3d0; }
    .insight-info  { background: #0a1f2d; border-left: 3px solid #60a5fa; color: #bfdbfe; }
    .insight-error { background: #2d0a0a; border-left: 3px solid #f87171; color: #fecaca; }

    /* Decision cards */
    .decision-card {
        background: #12151d;
        border: 1px solid #1e2535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .decision-stage {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .decision-action {
        font-weight: 600;
        font-size: 1rem;
        color: #60a5fa;
        margin: 2px 0;
    }
    .decision-reason { color: #94a3b8; font-size: 0.82rem; }

    /* Score bar */
    .score-bar-wrap { margin: 4px 0 0 0; }
    .score-bar-bg {
        background: #1e2535;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #12151d;
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
        border: 1px solid #1e2535;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #64748b;
        border-radius: 6px;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2535 !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar labels */
    .stTextInput label, .stSelectbox label, .stFileUploader label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        width: 100%;
        padding: 0.55rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #1e2535; border-radius: 8px; overflow: hidden; }

    /* Image grid */
    .plot-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #475569;
        text-align: center;
        margin-top: 4px;
    }

    /* Log area */
    .log-area {
        background: #080a0f;
        border: 1px solid #1e2535;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        max-height: 300px;
        overflow-y: auto;
    }
    .log-line { margin-bottom: 4px; }
    .log-line.success { color: #34d399; }
    .log-line.warn    { color: #fbbf24; }
    .log-line.error   { color: #f87171; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
defaults = {
    "stop": False,
    "running": False,
    "latest_state": {},
    "step_status": {},
    "step_times": {},
    "log_lines": [],
    "completed": False,
    "tuning_live": {},
    "loaded_cols": [],
    "loaded_path": "",
    "loaded_dtypes": {},
    "api_key_ok": None,       # None = not checked yet
    "api_key_msg": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Validate API key once per session ────────
if st.session_state.api_key_ok is None:
    with st.spinner("Validating OpenAI API key…"):
        _ok, _msg = validate_api_key()
    st.session_state.api_key_ok  = _ok
    st.session_state.api_key_msg = _msg

STEPS = [
    "profiling", "eda", "stats",
    "decision_pre", "preprocessing",
    "baseline", "decision_model",
    "tuning", "notebook"
]

STEP_LABELS = {
    "profiling":     "Data Profiling",
    "eda":           "Exploratory Analysis",
    "stats":         "Statistical Tests",
    "decision_pre":  "Preprocessing Decision",
    "preprocessing": "Feature Engineering",
    "baseline":      "Baseline Model",
    "decision_model":"Model Selection Decision",
    "tuning":        "Hyperparameter Tuning",
    "notebook":      "Notebook Generation",
}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem 0;'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;
                    background:linear-gradient(135deg,#60a5fa,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            ⚙️ Pipeline Config
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Data source ──────────────────────
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    manual_path   = st.text_input("— or enter file path", "train.csv")

    # Load columns whenever the source changes
    _src_key = (uploaded_file.name if uploaded_file else "", manual_path)
    if _src_key != (st.session_state.get("_last_src_key", ("", ""))):
        st.session_state["_last_src_key"] = _src_key
        try:
            if uploaded_file:
                _df_peek = pd.read_csv(uploaded_file, nrows=500)
                uploaded_file.seek(0)          # reset so it can be read again later
            elif manual_path and os.path.isfile(manual_path):
                _df_peek = pd.read_csv(manual_path, nrows=500)
            else:
                _df_peek = None

            if _df_peek is not None:
                st.session_state.loaded_cols   = list(_df_peek.columns)
                st.session_state.loaded_dtypes = {c: str(_df_peek[c].dtype) for c in _df_peek.columns}
                st.session_state.loaded_path   = (
                    f"uploaded_{uploaded_file.name}" if uploaded_file else manual_path
                )
        except Exception:
            st.session_state.loaded_cols   = []
            st.session_state.loaded_dtypes = {}

    # ── Step 2: Target column dropdown ──────────
    cols = st.session_state.loaded_cols
    if cols:
        # Keep previous selection if still valid
        prev_target = st.session_state.get("_sel_target", cols[-1])
        default_idx = cols.index(prev_target) if prev_target in cols else len(cols) - 1
        target_column = st.selectbox("Target Column", cols, index=default_idx)
        st.session_state["_sel_target"] = target_column
    else:
        target_column = st.text_input("Target Column", "target")
        st.session_state["_sel_target"] = target_column

    # ── Step 3: Auto-detect problem type ────────
    def _detect_problem_type(col_name: str) -> str:
        dtypes = st.session_state.loaded_dtypes
        df_cols = st.session_state.loaded_cols
        dtype = dtypes.get(col_name, "")
        if dtype in ("object", "bool") or "int" not in dtype and "float" not in dtype:
            return "classification"
        # numeric target: few unique values → classification
        if cols:
            try:
                if uploaded_file:
                    _tmp = pd.read_csv(uploaded_file, usecols=[col_name], nrows=5000)
                    uploaded_file.seek(0)
                elif manual_path and os.path.isfile(manual_path):
                    _tmp = pd.read_csv(manual_path, usecols=[col_name], nrows=5000)
                else:
                    _tmp = None
                if _tmp is not None and _tmp[col_name].nunique() <= 20:
                    return "classification"
            except Exception:
                pass
        return "regression"

    auto_type = _detect_problem_type(target_column) if cols else "classification"
    _type_label = {
        "classification": "Classification (auto-detected)",
        "regression":     "Regression (auto-detected)",
    }
    st.markdown(
        f"<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;"
        f"color:#60a5fa;margin:0.3rem 0 0.2rem 0;'>"
        f"Problem type: {_type_label[auto_type]}</div>",
        unsafe_allow_html=True,
    )
    _override = st.checkbox("Override problem type", value=False)
    if _override:
        problem_type = st.selectbox(
            "Problem Type",
            ["classification", "regression"],
            index=0 if auto_type == "classification" else 1,
        )
    else:
        problem_type = auto_type

    # dataset_path used downstream
    dataset_path = st.session_state.loaded_path or manual_path

    # ── API key status banner ────────────────────
    st.markdown("<hr style='border-color:#1e2535;margin:1rem 0;'>", unsafe_allow_html=True)
    _api_ok  = st.session_state.api_key_ok
    _api_msg = st.session_state.api_key_msg
    if _api_ok is True:
        st.markdown(
            f"<div style='font-size:0.72rem;font-family:JetBrains Mono,monospace;"
            f"color:#34d399;margin-bottom:0.6rem;'>✔ {_api_msg}</div>",
            unsafe_allow_html=True,
        )
    elif _api_ok is False:
        st.markdown(
            f"<div style='font-size:0.72rem;font-family:JetBrains Mono,monospace;"
            f"color:#f87171;margin-bottom:0.6rem;'>✘ {_api_msg}</div>",
            unsafe_allow_html=True,
        )

    col_run, col_stop = st.columns(2)
    with col_run:
        run_button = st.button("▶ Run", type="primary", disabled=(_api_ok is False))
    with col_stop:
        stop_button = st.button("⛔ Stop")

    if stop_button:
        st.session_state.stop = True

    sidebar_tracker = st.empty()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class='dashboard-header'>
    <h1 class='dashboard-title'>Agentic ML System</h1>
    <div class='dashboard-subtitle'>LangGraph · Multi-Agent Pipeline · Live Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper: render insight line
# ─────────────────────────────────────────────
def render_insight(text: str):
    if text.startswith("⚠️"):
        cls = "insight-warn"
    elif text.startswith("✅"):
        cls = "insight-good"
    elif text.startswith("❌"):
        cls = "insight-error"
    else:
        cls = "insight-info"
    st.markdown(f"<div class='insight-card {cls}'>{text}</div>", unsafe_allow_html=True)

def score_color(score):
    if score is None: return "info"
    if score >= 0.85: return "good"
    if score >= 0.70: return "warn"
    return "error"

def score_bar_html(score, color_class):
    color_map = {"good": "#34d399", "warn": "#fbbf24", "info": "#60a5fa", "error": "#f87171"}
    color = color_map.get(color_class, "#60a5fa")
    pct   = round((score or 0) * 100, 1)
    return f"""
    <div class='score-bar-wrap'>
        <div style='font-size:0.7rem;font-family:JetBrains Mono,monospace;color:#64748b;margin-bottom:3px;'>{pct}%</div>
        <div class='score-bar-bg'>
            <div class='score-bar-fill' style='width:{pct}%;background:{color};'></div>
        </div>
    </div>"""

def add_log(msg, level="info"):
    ts  = time.strftime("%H:%M:%S")
    st.session_state.log_lines.append({"ts": ts, "msg": msg, "level": level})


def render_live_logs(container):
    if not st.session_state.log_lines:
        container.info("No log entries yet.")
        return

    lines_html = ""
    for entry in st.session_state.log_lines:
        cls = entry.get("level", "info")
        lines_html += f"<div class='log-line {cls}'>[{entry['ts']}] {entry['msg']}</div>"
    container.markdown(f"<div class='log-area'>{lines_html}</div>", unsafe_allow_html=True)


def render_sidebar_tracker(container):
    with container.container():
        if not st.session_state.step_status:
            return

        st.markdown("<hr style='border-color:#1e2535;margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>Pipeline Steps</div>", unsafe_allow_html=True)

        rows_html = ""
        for step in STEPS:
            status = st.session_state.step_status.get(step, "pending")
            t = st.session_state.step_times.get(step, "")
            check_cls = {"done": "check-done", "running": "check-running", "pending": "check-pending"}.get(status, "check-pending")
            mark = {"done": "✓", "running": "•", "pending": ""}.get(status, "")
            name_cls = {"done": "done", "running": "running", "pending": ""}.get(status, "")
            time_str = f"{t:.1f}s" if isinstance(t, float) else ""
            rows_html += f"""
            <div class='step-row'>
                <div class='step-check {check_cls}'>{mark}</div>
                <div class='step-name {name_cls}'>{STEP_LABELS.get(step, step)}</div>
                <div class='step-time'>{time_str}</div>
            </div>"""

        st.markdown(f"<div class='pipeline-track'>{rows_html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────
tab_overview, tab_insights, tab_models, tab_decisions, tab_plots, tab_state, tab_log = st.tabs([
    "📊 Overview",
    "💡 Insights",
    "🏆 Models",
    "🧠 Decisions",
    "📈 Plots",
    "📦 State",
    "🖥️ Log",
])

with tab_overview:
    overview_slot = st.empty()
with tab_insights:
    insights_slot = st.empty()
with tab_models:
    models_slot = st.empty()
with tab_decisions:
    decisions_slot = st.empty()
with tab_plots:
    plots_slot = st.empty()
with tab_state:
    state_slot = st.empty()
with tab_log:
    log_slot = st.empty()
    clear_log_clicked = st.button("🗑️ Clear log")

if clear_log_clicked:
    st.session_state.log_lines = []
    st.rerun()


def render_overview(container):
    ls = st.session_state.latest_state
    with container.container():
        if not ls:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#3b4a6b;'>
                <div style='font-size:3rem;'>🧠</div>
                <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:600;margin-top:1rem;'>
                    Configure and run your pipeline
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;margin-top:0.5rem;'>
                    Set your dataset path and target column in the sidebar, then click ▶ Run
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        c1, c2, c3, c4 = st.columns(4)

        profile = ls.get("profiling_report", {})
        rows = profile.get("num_rows", "—")
        cols = profile.get("num_columns", "—")
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Dataset Shape</div>
                <div style='font-size:1.15rem;font-weight:700;font-family:JetBrains Mono,monospace;
                            color:#60a5fa;line-height:1.2;margin-top:2px;'>{rows} × {cols}</div>
            </div>""", unsafe_allow_html=True)

        b_score = ls.get("baseline_result", {}).get("score")
        b_cls = score_color(b_score)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Baseline Score</div>
                <div class='metric-value {b_cls}'>{f"{b_score:.4f}" if b_score is not None else "—"}</div>
                {score_bar_html(b_score, b_cls) if b_score is not None else ""}
            </div>""", unsafe_allow_html=True)

        a_score = ls.get("advanced_result", {}).get("score")
        a_cls = score_color(a_score)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Advanced Score</div>
                <div class='metric-value {a_cls}'>{f"{a_score:.4f}" if a_score is not None else "—"}</div>
                {score_bar_html(a_score, a_cls) if a_score is not None else ""}
            </div>""", unsafe_allow_html=True)

        done_count = sum(1 for v in st.session_state.step_status.values() if v == "done")
        total = len(STEPS)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Steps Completed</div>
                <div class='metric-value info'>{done_count} / {total}</div>
                {score_bar_html(done_count / total if total else 0, "info")}
            </div>""", unsafe_allow_html=True)

        missing = ls.get("eda_report", {}).get("missing_summary", {})
        if missing:
            st.markdown("<div style='margin-top:1.5rem;font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;'>Missing Values</div>", unsafe_allow_html=True)
            miss_df = pd.DataFrame(missing.items(), columns=["Column", "Missing %"]).sort_values("Missing %", ascending=False)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

        corr_data = ls.get("eda_report", {}).get("correlation", {}).get("target_correlations", {})
        if corr_data:
            st.markdown("<div style='margin-top:1.5rem;font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;'>Feature–Target Correlations</div>", unsafe_allow_html=True)
            corr_df = pd.DataFrame(corr_data.items(), columns=["Feature", "|r| with Target"]).sort_values("|r| with Target", ascending=False)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)


def render_insights_tab(container):
    ls = st.session_state.latest_state
    eda_report = ls.get("eda_report", {})
    insights = eda_report.get("insights", [])

    with container.container():
        if not insights:
            st.info("No insights yet — run the pipeline first.")
            return

        warn_list = [i for i in insights if i.startswith("⚠️")]
        good_list = [i for i in insights if i.startswith("✅")]
        error_list = [i for i in insights if i.startswith("❌")]
        info_list = [i for i in insights if not any(i.startswith(p) for p in ["⚠️", "✅", "❌"])]

        if warn_list or error_list:
            st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>⚠️ Warnings ({len(warn_list) + len(error_list)})</div>", unsafe_allow_html=True)
            for item in error_list + warn_list:
                render_insight(item)

        if good_list:
            st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#34d399;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem 0;'>✅ Good Signals ({len(good_list)})</div>", unsafe_allow_html=True)
            for item in good_list:
                render_insight(item)

        if info_list:
            st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem 0;'>ℹ️ Info ({len(info_list)})</div>", unsafe_allow_html=True)
            for item in info_list:
                render_insight(item)


def render_live_tuning_status():
    live = st.session_state.get("tuning_live", {})
    if not live:
        return

    st.markdown("**Live Tuning Status**")
    current = live.get("current", {})
    summary_rows = [
        {"Field": "Phase", "Value": current.get("phase", "N/A")},
        {"Field": "Model", "Value": current.get("label", current.get("model_id", "N/A"))},
        {"Field": "Strategy", "Value": current.get("strategy", "N/A")},
        {"Field": "Metric", "Value": current.get("metric", "N/A")},
        {"Field": "Value", "Value": current.get("holdout_primary_score", current.get("cv_score", current.get("score", "N/A")))},
        {"Field": "Params", "Value": json.dumps(current.get("params", {}), default=str)},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    history = live.get("history", [])
    if history:
        st.markdown("**Recent Tuning Events**")
        hist_df = pd.DataFrame(history[-12:])
        preferred_cols = [
            "phase",
            "label",
            "strategy",
            "trial_index",
            "trial_count",
            "metric",
            "cv_score",
            "holdout_primary_score",
            "score",
            "params",
            "error",
        ]
        available_cols = [col for col in preferred_cols if col in hist_df.columns]
        if available_cols:
            hist_df = hist_df[available_cols]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


def render_models_tab(container):
    ls = st.session_state.latest_state
    b_result = ls.get("baseline_result", {})
    a_result = ls.get("advanced_result", {})

    with container.container():
        if not b_result and not a_result:
            if st.session_state.get("tuning_live"):
                render_live_tuning_status()
            else:
                st.info("No model results yet.")
            return

        if st.session_state.get("tuning_live"):
            render_live_tuning_status()

        col_b, col_a = st.columns(2)
        for col, result, label in [(col_b, b_result, "Baseline"), (col_a, a_result, "Advanced")]:
            with col:
                if result:
                    score = result.get("score")
                    cls = score_color(score)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label} Model</div>
                        <div class='metric-value {cls}'>{f"{score:.4f}" if score is not None else "—"}</div>
                        {score_bar_html(score, cls) if score is not None else ""}
                    </div>""", unsafe_allow_html=True)

                    display = {
                        k: v for k, v in result.items()
                        if k not in {"score", "imbalance_comparison", "candidate_results"}
                    }
                    if display:
                        rows = [{"Metric": k, "Value": str(v)} for k, v in display.items()]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    render_imbalance_comparison(result)
                    render_candidate_results(result)
                else:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div style='color:#3b4a6b;font-size:0.82rem;'>Not run</div></div>", unsafe_allow_html=True)


def render_imbalance_comparison(result):
    comparison = result.get("imbalance_comparison", {})
    if not comparison:
        return

    st.markdown("**Imbalance Handling Comparison**")

    if not comparison.get("tested"):
        st.caption("Imbalance handling was not evaluated in this run because the target did not cross the imbalance threshold.")
        return

    rows = []
    scenarios = [
        ("without_imbalance_handling", "Without imbalance handling"),
        ("with_imbalance_handling", "With class_weight=balanced"),
    ]
    for key, label in scenarios:
        item = comparison.get(key)
        if not item:
            continue
        rows.append({
            "Scenario": label,
            "Best Model": item.get("model", "N/A"),
            "Metric": item.get("metric", "N/A"),
            "Score": item.get("score"),
            "Accuracy": item.get("accuracy"),
            "Weighted F1": item.get("f1_weighted"),
            "Selected": "Yes" if item.get("selected") else "No",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected_strategy = comparison.get("selected_strategy", "unweighted")
    delta = comparison.get("score_delta_balanced_minus_unweighted")
    if delta is not None:
        impact = "improved" if delta > 0 else "reduced" if delta < 0 else "did not change"
        st.caption(
            f"Selected strategy: `{selected_strategy}`. Balanced minus unweighted score delta: {delta:+.4f}, which {impact} performance."
        )
    else:
        st.caption(f"Selected strategy: `{selected_strategy}`.")


def render_candidate_results(result):
    candidates = result.get("candidate_results", [])
    if not candidates:
        return

    st.markdown("**Candidate Results**")
    candidate_df = pd.DataFrame(candidates)
    preferred_cols = [
        "model_id",
        "model",
        "strategy",
        "imbalance_strategy",
        "cv_score",
        "score",
        "accuracy",
        "f1_weighted",
        "r2",
        "mae",
        "rmse",
        "best_params",
        "note",
    ]
    available_cols = [col for col in preferred_cols if col in candidate_df.columns]
    if available_cols:
        candidate_df = candidate_df[available_cols]
    st.dataframe(candidate_df, use_container_width=True, hide_index=True)


def render_decisions_tab(container):
    ls = st.session_state.latest_state
    decision_log = ls.get("decision_log", {})

    with container.container():
        if not decision_log:
            st.info("No decisions logged yet.")
            return

        for stage, info in decision_log.items():
            action = info.get("action", "—")
            reason = info.get("reason", "")
            llm_error = info.get("llm_error")
            st.markdown(f"""
            <div class='decision-card'>
                <div class='decision-stage'>{stage}</div>
                <div class='decision-action'>→ {action}</div>
                <div class='decision-reason'>{reason}</div>
            </div>""", unsafe_allow_html=True)
            if llm_error:
                st.caption(f"LLM error: {llm_error}")


def render_plots_tab(container):
    ls = st.session_state.latest_state
    plots = ls.get("eda_report", {}).get("plots", [])

    with container.container():
        if not plots:
            st.info("No plots generated yet.")
            return

        filter_opts = ["All", "Distribution", "Boxplot", "Correlation", "Target", "Missing"]
        if st.session_state.running:
            chosen = st.session_state.get("plot_filter", "All")
        else:
            chosen = st.selectbox("Filter plots", filter_opts, key="plot_filter")

        filter_map = {
            "Distribution": "_analysis",
            "Boxplot": "_box",
            "Correlation": "correlation",
            "Target": "target_",
            "Missing": "missing",
        }
        keyword = filter_map.get(chosen, "")
        filtered_plots = [p for p in plots if not keyword or keyword in p]

        if not filtered_plots:
            st.info("No plots match this filter.")
            return

        for i in range(0, len(filtered_plots), 2):
            row_cols = st.columns(2)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx < len(filtered_plots):
                    p = filtered_plots[idx]
                    if os.path.exists(p):
                        with col:
                            st.image(p, use_container_width=True)
                            st.markdown(f"<div class='plot-caption'>{Path(p).stem}</div>", unsafe_allow_html=True)


def render_state_tab(container):
    ls = st.session_state.latest_state
    with container.container():
        if not ls:
            st.info("No state yet.")
            return

        skip_keys = {"raw_data", "processed_data"}
        rows = []
        for k, v in ls.items():
            if k in skip_keys:
                continue
            if isinstance(v, dict):
                val_str = json.dumps(v, default=str)[:300]
            else:
                val_str = str(v)[:300]
            rows.append({"Key": k, "Type": type(v).__name__, "Preview": val_str})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("🔍 Full state (JSON)"):
            safe = {k: str(v)[:500] for k, v in ls.items() if k not in skip_keys}
            st.code(json.dumps(safe, indent=2, default=str), language="json")


def render_log_tab(container):
    with container.container():
        render_live_logs(st)


def render_dashboard():
    render_sidebar_tracker(sidebar_tracker)
    render_overview(overview_slot)
    render_insights_tab(insights_slot)
    render_models_tab(models_slot)
    render_decisions_tab(decisions_slot)
    render_plots_tab(plots_slot)
    render_state_tab(state_slot)
    render_log_tab(log_slot)


render_dashboard()

# ─────────────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────────────
if run_button:
    # Save uploaded file to disk if provided (sidebar already set dataset_path)
    if uploaded_file:
        save_path = f"uploaded_{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        dataset_path = save_path

    # Auto-generate objective from problem type and target
    objective = (
        f"Maximize {'accuracy' if problem_type == 'classification' else 'R²'} "
        f"for target '{target_column}'"
    )

    # Reset session
    st.session_state.stop      = False
    st.session_state.running   = True
    st.session_state.completed = False
    st.session_state.latest_state = {}
    st.session_state.log_lines = []
    st.session_state.tuning_live = {}
    st.session_state.step_status = {s: "pending" for s in STEPS}
    st.session_state.step_times  = {}
    st.session_state.step_status["profiling"] = "running"

    graph = build_graph()

    def progress_callback(event):
        history = list(st.session_state.tuning_live.get("history", []))
        history.append(event)
        st.session_state.tuning_live = {
            "current": event,
            "history": history[-100:],
        }
        st.session_state.latest_state["tuning_live"] = st.session_state.tuning_live

        phase = event.get("phase")
        label = event.get("label", event.get("model_id", "model"))
        if phase == "start":
            add_log(
                f"Tuning stage started with {len(event.get('models', []))} candidate models, metric={event.get('metric')}, cv_folds={event.get('cv_folds')}, optuna_trials={event.get('optuna_trials_per_model')}.",
                "info",
            )
        elif phase == "start_model":
            add_log(
                f"Running {label} ({event.get('model_index')}/{event.get('model_count')}) with strategies={event.get('strategies')}.",
                "info",
            )
        elif phase == "trial_complete":
            add_log(
                f"{label} trial {event.get('trial_index')}/{event.get('trial_count')} [{event.get('strategy')}] cv={event.get('cv_score'):.4f}, holdout={event.get('holdout_primary_score'):.4f}.",
                "info",
            )
        elif phase == "best_update":
            add_log(
                f"New best tuned model: {label} [{event.get('strategy')}] cv={event.get('cv_score'):.4f}, holdout={event.get('holdout_primary_score'):.4f}.",
                "success",
            )
        elif phase == "trial_error":
            add_log(f"{label} trial failed: {event.get('error')}", "warn")
        elif phase == "completed":
            add_log(
                f"Tuning completed. Best model={label}, strategy={event.get('strategy')}, score={event.get('score'):.4f}.",
                "success",
            )

        render_dashboard()

    initial_state = {
        "dataset_path":  dataset_path,
        "target_column": target_column,
        "problem_type":  problem_type,
        "objective":     objective,
        "_progress_callback": progress_callback,
    }

    add_log(f"Pipeline started — target='{target_column}', type='{problem_type}'", "info")
    render_dashboard()

    step_start = {"profiling": time.time()}

    for chunk in graph.stream(initial_state):
        if st.session_state.stop:
            add_log("Pipeline stopped by user.", "warn")
            render_dashboard()
            st.warning("⛔ Pipeline stopped by user.")
            break

        node_name   = list(chunk.keys())[0]
        node_output = chunk[node_name]
        now         = time.time()

        # Timing
        if node_name not in step_start:
            step_start[node_name] = now
        elapsed = now - step_start.get(node_name, now)

        # Update tracking
        st.session_state.step_status[node_name] = "done"
        st.session_state.step_times[node_name]  = elapsed
        st.session_state.latest_state.update(node_output)
        if node_output.get("error"):
            error_msg = node_output["error"]
            st.session_state.running = False
            st.session_state.completed = False
            add_log(f"Pipeline failed in {STEP_LABELS.get(node_name, node_name)}: {error_msg}", "error")
            render_dashboard()
            st.error(error_msg)
            break

        add_log(f"✅ {STEP_LABELS.get(node_name, node_name)} completed ({elapsed:.1f}s)", "success")

        # Mark next step as running
        idx = STEPS.index(node_name) if node_name in STEPS else -1
        if idx >= 0 and idx + 1 < len(STEPS):
            next_step = STEPS[idx + 1]
            if st.session_state.step_status.get(next_step) == "pending":
                st.session_state.step_status[next_step] = "running"
                step_start[next_step] = time.time()

        render_dashboard()

    st.session_state.running = False
    if not st.session_state.stop and not st.session_state.latest_state.get("error"):
        st.session_state.completed = True
        add_log("Pipeline completed.", "success")
    render_dashboard()

# ─────────────────────────────────────────────
ls = st.session_state.latest_state

# Completion Banner
if st.session_state.completed:
    total_time = sum(v for v in st.session_state.step_times.values() if isinstance(v, float))
    st.success(f"✅ Pipeline completed in {total_time:.1f}s")
    notebook_path = ls.get("notebook_path", "output_notebook.ipynb")
    if os.path.exists(str(notebook_path)):
        with open(notebook_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Notebook",
                data=f,
                file_name=os.path.basename(str(notebook_path)),
                mime="application/octet-stream",
            )
