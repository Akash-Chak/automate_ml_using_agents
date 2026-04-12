# main.py

from orchestrator.graph import run_pipeline

initial_state = {
    "dataset_path": "data.csv",
    "target_column": "target",
    "problem_type": "classification",
    "objective": "Maximize accuracy"
}

final_state = run_pipeline(initial_state)

print(final_state["advanced_result"])