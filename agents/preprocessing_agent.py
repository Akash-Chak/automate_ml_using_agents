# agents/preprocessing_agent.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing_agent(state):
    df = state["raw_data"].copy()
    target = state["target_column"]

    # ✅ 1. Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # ✅ 2. Separate features & target
    X = df.drop(columns=[target])
    y = df[target]

    # ✅ 3. Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # ✅ 4. Encode categorical variables
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ✅ 5. Scale numerical features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # ✅ 6. Recombine
    df_processed = pd.concat([X, y], axis=1)

    state["processed_data"] = df_processed
    state["preprocessing_report"] = {
        "steps": [
            "filled_missing_values",
            "one_hot_encoding",
            "feature_scaling"
        ],
        "num_features": len(num_cols),
        "cat_features": len(cat_cols)
    }

    return state