# agents/preprocessing_agent.py

from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocessing_agent(state):
    df = state["raw_data"]

    df = df.dropna()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    state["processed_data"] = df
    state["preprocessing_report"] = {
        "steps": ["dropna", "scaling"]
    }

    return state