# agents/profiling_agent.py

import pandas as pd

def profiling_agent(state):
    df = pd.read_csv(state["dataset_path"])

    profile = {}

    profile["num_rows"] = df.shape[0]
    profile["num_columns"] = df.shape[1]

    column_summary = {}

    for col in df.columns:
        col_data = df[col]

        column_summary[col] = {
            "dtype": str(col_data.dtype),
            "missing_pct": col_data.isnull().mean(),
            "cardinality": col_data.nunique(),
            "is_numeric": pd.api.types.is_numeric_dtype(col_data),
            "potential_outliers": (
                col_data.skew() > 1 if pd.api.types.is_numeric_dtype(col_data) else False
            )
        }

    profile["column_summary"] = column_summary

    state["raw_data"] = df
    state["profiling_report"] = profile

    return state