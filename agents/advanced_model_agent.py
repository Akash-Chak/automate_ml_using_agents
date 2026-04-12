# agents/advanced_model_agent.py

from utils.data_utils import ensure_processed_data

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def advanced_model_agent(state):

    target = state["target_column"]

    # ✅ Use centralized preprocessing
    df = ensure_processed_data(state)

    X = df.drop(columns=[target])
    y = df[target]

    # -------------------------------
    # Train-Test Split (NO leakage)
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Model Selection
    # -------------------------------
    if state["problem_type"] == "classification":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)

    else:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = mean_absolute_error(y_test, preds)

    # -------------------------------
    # Store Results
    # -------------------------------
    state["advanced_result"] = {
        "model": type(model).__name__,
        "score": float(score)
    }

    return state