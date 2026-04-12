# agents/advanced_model_agent.py

from sklearn.ensemble import RandomForestClassifier

def advanced_model_agent(state):
    df = state["processed_data"]
    target = state["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    model = RandomForestClassifier()
    model.fit(X, y)

    score = model.score(X, y)

    state["advanced_result"] = {
        "model": "RandomForest",
        "score": score
    }

    return state