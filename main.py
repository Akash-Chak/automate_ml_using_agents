# main.py
from dotenv import load_dotenv
load_dotenv()
from orchestrator.langgraph_pipeline import build_graph

initial_state = {
    "dataset_path": "data.csv",
    "target_column": "target",
    "problem_type": "classification",
    "objective": "Maximize prediction accuracy"
}

graph = build_graph()
graph.get_graph(xray=True).draw_png("graph.png")
final_state = graph.invoke(initial_state)

print("✅ Pipeline Completed")
print(final_state["advanced_result"])