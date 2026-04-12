from utils.data_utils import build_preprocessed_dataset


def preprocessing_agent(state):
    if state.get("error"):
        return state

    try:
        processed_data, preprocessing_report = build_preprocessed_dataset(state)
    except Exception as exc:
        return {**state, "error": f"Preprocessing failed: {exc}"}

    state["processed_data"] = processed_data
    state["preprocessing_report"] = preprocessing_report
    return state
