# orchestrator/langgraph_pipeline.py

from langgraph.graph import StateGraph, END

from agents.profiling_agent import profiling_agent
from agents.eda_agent import eda_agent
from agents.stats_agent import stats_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.baseline_model_agent import baseline_model_agent
from agents.advanced_model_agent import advanced_model_agent
from agents.decision_agent import decision_agent
from agents.notebook_agent import notebook_agent
from state import AgentState


def build_graph():

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("profiling", profiling_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("stats", stats_agent)

    graph.add_node("decision_pre", lambda s: decision_agent(s, "preprocessing"))
    graph.add_node("preprocessing", preprocessing_agent)

    graph.add_node("baseline", baseline_model_agent)
    graph.add_node("decision_model", lambda s: decision_agent(s, "model_selection"))

    graph.add_node("advanced", advanced_model_agent)
    graph.add_node("notebook", notebook_agent)

    # Flow
    graph.set_entry_point("profiling")

    graph.add_edge("profiling", "eda")
    graph.add_edge("eda", "stats")

    graph.add_edge("stats", "decision_pre")

    # Conditional routing
    def route_preprocessing(state):
        action = state["decision_log"]["preprocessing"]["action"]
        return "preprocessing" if action != "skip" else "baseline"

    graph.add_conditional_edges(
        "decision_pre",
        route_preprocessing,
        {
            "preprocessing": "preprocessing",
            "baseline": "baseline"
        }
    )

    graph.add_edge("preprocessing", "baseline")
    graph.add_edge("baseline", "decision_model")

    # Model decision routing
    def route_model(state):
        score = state["baseline_result"]["score"]

        if score > 0.8:
            return "notebook"
        else:
            return "advanced"

    graph.add_conditional_edges(
        "decision_model",
        route_model,
        {
            "advanced": "advanced",
            "notebook": "notebook"
        }
    )

    graph.add_edge("advanced", "notebook")
    graph.add_edge("notebook", END)

    return graph.compile()