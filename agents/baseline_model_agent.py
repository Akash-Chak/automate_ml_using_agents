# agents/baseline_model_agent.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def baseline_model_agent(state):
    df = state["processed_data"]
    target = state["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if state["problem_type"] == "classification":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = accuracy_score(y_test, preds)

    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = mean_absolute_error(y_test, preds)

    state["baseline_result"] = {
        "score": score
    }

    return state