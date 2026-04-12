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

    # callback injected by the Streamlit app; not serialised — kept as-is by LangGraph
    _progress_callback: Optional[Callable]
