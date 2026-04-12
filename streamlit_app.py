# streamlit_app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from orchestrator.langgraph_pipeline import build_graph

# -------------------------------
# UI Setup
# -------------------------------
st.set_page_config(layout="wide")
st.title("🧠 Agentic ML System - Live Dashboard")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Configuration")

dataset_path = st.sidebar.text_input("Dataset Path", "train.csv")
target_column = st.sidebar.text_input("Target Column", "target")
problem_type = st.sidebar.selectbox("Problem Type", ["classification", "regression"])
objective = st.sidebar.text_input("Objective", "Maximize accuracy")

run_button = st.sidebar.button("🚀 Run Pipeline")
stop_button = st.sidebar.button("⛔ Stop Pipeline")

# -------------------------------
# Session State
# -------------------------------
if "stop" not in st.session_state:
    st.session_state.stop = False

if stop_button:
    st.session_state.stop = True

# -------------------------------
# Layout
# -------------------------------
progress_bar = st.progress(0)
status_text = st.empty()

step_container = st.container()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Metrics",
    "🧠 Decisions",
    "📦 State",
    "📈 Plots"
])

# -------------------------------
# Run Pipeline
# -------------------------------
if run_button:

    st.session_state.stop = False

    graph = build_graph()

    initial_state = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "problem_type": problem_type,
        "objective": objective
    }

    steps_order = [
        "profiling",
        "eda",
        "stats",
        "decision_pre",
        "preprocessing",
        "baseline",
        "decision_model",
        "advanced",
        "notebook"
    ]

    step_status = {step: "pending" for step in steps_order}

    latest_state = {}

    # -------------------------------
    # Stream Execution
    # -------------------------------
    for i, step in enumerate(graph.stream(initial_state)):

        if st.session_state.stop:
            st.warning("⛔ Pipeline Stopped")
            break

        node_name = list(step.keys())[0]
        node_output = step[node_name]

        latest_state.update(node_output)

        # Update status
        step_status[node_name] = "done"

        # -------------------------------
        # Progress Bar
        # -------------------------------
        progress = (i + 1) / len(steps_order)
        progress_bar.progress(min(progress, 1.0))

        status_text.markdown(f"### 🚀 Running: `{node_name}`")

        # -------------------------------
        # Step Tracker UI
        # -------------------------------
        with step_container:
            st.subheader("🔄 Pipeline Steps")

            for step_name in steps_order:
                if step_status[step_name] == "done":
                    st.write(f"✅ {step_name}")
                elif step_name == node_name:
                    st.write(f"⏳ {step_name}")
                else:
                    st.write(f"⬜ {step_name}")

        # -------------------------------
        # 📊 Metrics Tab
        # -------------------------------
        with tab1:
            metrics = []

            if "baseline_result" in latest_state:
                metrics.append({
                    "Model": "Baseline",
                    "Score": latest_state["baseline_result"].get("score")
                })

            if "advanced_result" in latest_state:
                metrics.append({
                    "Model": "Advanced",
                    "Score": latest_state["advanced_result"].get("score")
                })

            if metrics:
                df_metrics = pd.DataFrame(metrics)
                st.dataframe(df_metrics, use_container_width=True)

        # -------------------------------
        # 🧠 Decisions Tab
        # -------------------------------
        with tab2:
            if "decision_log" in latest_state:
                decisions = latest_state["decision_log"]

                rows = []
                for stage, info in decisions.items():
                    rows.append({
                        "Stage": stage,
                        "Action": info.get("action"),
                        "Reason": info.get("reason", "")
                    })

                df_decisions = pd.DataFrame(rows)
                st.dataframe(df_decisions, use_container_width=True)

        # -------------------------------
        # 📦 State Tab
        # -------------------------------
        with tab3:
            filtered_state = {
                k: str(v)[:200]
                for k, v in latest_state.items()
                if k not in ["raw_data", "processed_data"]
            }

            df_state = pd.DataFrame(filtered_state.items(), columns=["Key", "Value"])
            st.dataframe(df_state, use_container_width=True)

        # -------------------------------
        # 📈 Plots Tab
        # -------------------------------
        with tab4:
            plots = latest_state.get("eda_report", {}).get("plots", [])

            if plots:
                for p in plots[:10]:  # limit for performance
                    st.image(p)
            else:
                st.info("No plots generated yet")

    progress_bar.progress(1.0)
    status_text.markdown("### ✅ Pipeline Completed")