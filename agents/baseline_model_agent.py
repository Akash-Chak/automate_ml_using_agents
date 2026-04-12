# agents/baseline_model_agent.py

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from utils.data_utils import ensure_processed_data

def baseline_model_agent(state):

    target = state["target_column"]

    df = ensure_processed_data(state)

    # -------------------------------
    # Model Training
    # -------------------------------
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if state["problem_type"] == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = accuracy_score(y_test, preds)

    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = mean_absolute_error(y_test, preds)

    # -------------------------------
    # Store results
    # -------------------------------
    state["baseline_result"] = {
        "model": type(model).__name__,
        "score": float(score)  # ensure JSON safe
    }

    return state