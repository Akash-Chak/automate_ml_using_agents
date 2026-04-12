# utils/data_utils.py

import pandas as pd

def ensure_processed_data(state):

    # ✅ If already exists → reuse
    if "processed_data" in state:
        return state["processed_data"]

    print("⚠️ processed_data missing → auto-generating")

    df = state["raw_data"].copy()
    target = state["target_column"]

    # -------------------------------
    # Minimal preprocessing fallback
    # -------------------------------

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Split features
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Recombine
    df_processed = pd.concat([X, y], axis=1)

    # Save to state
    state["processed_data"] = df_processed

    return df_processed