# agents/notebook_agent.py

import nbformat as nbf

from utils.markdown_formatter import (
    format_eda_report,
    format_stats_report,
    format_preprocessing,
    format_model_result,
    format_decision
)

def notebook_agent(state):

    nb = nbf.v4.new_notebook()

    def md(text):
        return nbf.v4.new_markdown_cell(text)

    def code(text):
        return nbf.v4.new_code_cell(text)

    # ✅ Safe getters (avoid None crashes)
    eda_report = state.get("eda_report", {})
    stats_report = state.get("stats_report", {})
    preprocessing_report = state.get("preprocessing_report", {})
    baseline_result = state.get("baseline_result", {})
    advanced_result = state.get("advanced_result", {})
    decision_log = state.get("decision_log", {})

    nb.cells = [

        # -------------------------------
        # 📌 Problem
        # -------------------------------
        md("# 📌 Problem Statement"),
        md(state.get("objective", "No objective provided")),

        # -------------------------------
        # 📦 Data Loading
        # -------------------------------
        md("# 📦 Data Loading"),
        code(f"""
import pandas as pd

df = pd.read_csv('{state["dataset_path"]}')
df.head()
"""),

        # -------------------------------
        # 🔍 Profiling
        # -------------------------------
        md("# 🔍 Data Profiling"),
        code("""
df.info()
df.describe()
"""),

        # -------------------------------
        # 📊 EDA
        # -------------------------------
        md("# 📊 Exploratory Data Analysis"),
        md(format_eda_report(eda_report)),

        # -------------------------------
        # 📈 Stats
        # -------------------------------
        md("# 📈 Statistical Analysis"),
        md(format_stats_report(stats_report)),

        # -------------------------------
        # 🧹 Preprocessing
        # -------------------------------
        md("# 🧹 Preprocessing"),
        md(format_preprocessing(preprocessing_report)),

        # -------------------------------
        # 🤖 Baseline
        # -------------------------------
        md("# 🤖 Baseline Model"),
        md(format_model_result(baseline_result, "Baseline Model")),

        # -------------------------------
        # 🚀 Advanced
        # -------------------------------
        md("# 🚀 Advanced Model"),
        md(format_model_result(advanced_result, "Advanced Model")),

        # -------------------------------
        # 🏆 Decision
        # -------------------------------
        md("# 🏆 Final Decision"),
        md(format_decision(decision_log))
    ]

    # ✅ Write notebook
    output_path = "output_notebook.ipynb"
    nbf.write(nb, output_path)

    print(f"📓 Notebook generated at: {output_path}")

    state["notebook_json"] = nb

    return state