# state.py

from typing import TypedDict, Dict, Any, Callable, Optional

class AgentState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    problem_type: str
    objective: str

    raw_data: Any
    processed_data: Any

    profiling_report: Dict
    eda_report: Dict
    stats_report: Dict
    preprocessing_report: Dict

    baseline_result: Dict
    advanced_result: Dict
    tuning_live: Dict

    decision_log: Dict
    notebook_json: Dict

    # MLflow experiment config (injected by Streamlit before pipeline starts)
    mlflow_experiment_name: Optional[str]   # e.g. "train_Irrigation_Need_classification"
    mlflow_tracking_uri:    Optional[str]   # default "./mlruns"
    tuning_mode:            Optional[str]   # "smoke_test" | "reuse_mlflow" | "full_search"

    # MLflow run IDs populated by agents
    mlflow_baseline_run_id: Optional[str]
    mlflow_hpo_run_id:      Optional[str]
    mlflow_experiment_url:  Optional[str]   # clickable link shown in UI

    # callback injected by the Streamlit app; not serialised — kept as-is by LangGraph
    _progress_callback: Optional[Callable]
