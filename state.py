# state.py

from typing import TypedDict, Dict, Any

class AgentState(TypedDict):
    dataset_path: str
    target_column: str
    problem_type: str
    objective: str

    raw_data: Any

    profiling_report: Dict
    eda_report: Dict
    stats_report: Dict
    preprocessing_report: Dict

    baseline_result: Dict
    advanced_result: Dict

    decision_log: Dict
    notebook_json: Dict