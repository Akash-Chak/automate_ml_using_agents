# agents/eda_agent.py

import matplotlib.pyplot as plt
import seaborn as sns

def eda_agent(state):
    df = state["raw_data"]
    target = state["target_column"]

    insights = []
    plots = []

    for col in df.columns:
        if col == target:
            continue

        if df[col].dtype == "object":
            insights.append(f"{col} is categorical")

        else:
            insights.append(f"{col} mean = {df[col].mean()}")

    state["eda_report"] = {
        "insights": insights,
        "plots": plots
    }

    return state