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

    tuning_mode:            Optional[str]   # "smoke_test" | "full_search"

    # callback injected by the UI; not serialised — kept as-is by LangGraph
    _progress_callback: Optional[Callable]
