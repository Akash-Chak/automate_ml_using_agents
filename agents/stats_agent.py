# agents/stats_agent.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway


def stats_agent(state):

    df = state["raw_data"]
    target = state["target_column"]

    significant = []
    insignificant = []
    test_details = {}

    target_values = df[target].dropna().unique()

    for col in df.columns:
        if col == target:
            continue

        try:
            # -------------------------------
            # CATEGORICAL vs TARGET
            # -------------------------------
            if df[col].dtype == "object" or df[col].dtype.name == "category":

                contingency = pd.crosstab(df[col], df[target])

                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue

                chi2, p, _, _ = chi2_contingency(contingency)

                test_details[col] = {
                    "type": "chi_square",
                    "p_value": float(p)
                }

            # -------------------------------
            # NUMERICAL vs TARGET (classification)
            # -------------------------------
            elif pd.api.types.is_numeric_dtype(df[col]):

                groups = [
                    df[df[target] == val][col].dropna()
                    for val in target_values
                ]

                # Need at least 2 groups with data
                if len(groups) < 2 or any(len(g) == 0 for g in groups):
                    continue

                stat, p = f_oneway(*groups)

                test_details[col] = {
                    "type": "anova",
                    "p_value": float(p)
                }

            else:
                continue

            # -------------------------------
            # SIGNIFICANCE DECISION
            # -------------------------------
            if test_details[col]["p_value"] < 0.05:
                significant.append(col)
            else:
                insignificant.append(col)

        except Exception as e:
            print(f"⚠️ Stats failed for {col}: {e}")
            continue

    # -------------------------------
    # Save output
    # -------------------------------
    state["stats_report"] = {
        "significant": significant,
        "insignificant": insignificant,
        "details": test_details
    }

    return state