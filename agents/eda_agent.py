# agents/eda_agent.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def eda_agent(state):

    df = state["raw_data"]
    target = state["target_column"]

    insights = []
    plots = []
    feature_summary = {}

    os.makedirs("outputs/eda", exist_ok=True)

    for col in df.columns:
        if col == target:
            continue

        col_info = {}

        # -------------------------------
        # Numerical Columns
        # -------------------------------
        if pd.api.types.is_numeric_dtype(df[col]):

            mean_val = df[col].mean()
            std_val = df[col].std()
            skew_val = df[col].skew()

            col_info = {
                "type": "numerical",
                "mean": float(mean_val),
                "std": float(std_val),
                "skew": float(skew_val)
            }

            insights.append(f"{col}: mean={mean_val:.2f}, skew={skew_val:.2f}")

            # 📊 Histogram
            plt.figure()
            sns.histplot(df[col], kde=True)
            plot_path = f"outputs/eda/{col}_hist.png"
            plt.savefig(plot_path)
            plt.close()
            plots.append(plot_path)

            # 📊 Boxplot (outlier view)
            plt.figure()
            sns.boxplot(x=df[col])
            plot_path = f"outputs/eda/{col}_box.png"
            plt.savefig(plot_path)
            plt.close()
            plots.append(plot_path)

        # -------------------------------
        # Categorical Columns
        # -------------------------------
        else:

            nunique = df[col].nunique()
            top_values = df[col].value_counts().head(3).to_dict()

            col_info = {
                "type": "categorical",
                "unique_values": int(nunique),
                "top_values": top_values
            }

            insights.append(f"{col}: {nunique} unique values")

            # 📊 Countplot
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df[col])
            plt.xticks(rotation=45)
            plot_path = f"outputs/eda/{col}_count.png"
            plt.savefig(plot_path)
            plt.close()
            plots.append(plot_path)

        feature_summary[col] = col_info

    # -------------------------------
    # Correlation (numerical only)
    # -------------------------------
    num_df = df.select_dtypes(include=np.number)

    if len(num_df.columns) > 1:
        corr = num_df.corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm")
        plot_path = "outputs/eda/correlation_heatmap.png"
        plt.savefig(plot_path)
        plt.close()

        plots.append(plot_path)

    # -------------------------------
    # Save everything
    # -------------------------------
    state["eda_report"] = {
        "insights": insights,
        "plots": plots,
        "feature_summary": feature_summary
    }

    return state