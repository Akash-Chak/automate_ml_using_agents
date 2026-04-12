# orchestrator/langgraph_pipeline.py

from langgraph.graph import StateGraph, END

from agents.profiling_agent import profiling_agent
from agents.eda_agent import eda_agent
from agents.stats_agent import stats_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.baseline_model_agent import baseline_model_agent
from agents.hyperparameter_tuning_agent import hyperparameter_tuning_agent
from agents.decision_agent import decision_agent
from agents.notebook_agent import notebook_agent
from state import AgentState


def build_graph():

    graph = StateGraph(AgentState)

    def route_on_error(state, next_node):
        return END if state.get("error") else next_node

    # Nodes
    graph.add_node("profiling", profiling_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("stats", stats_agent)

    graph.add_node("decision_pre", lambda s: decision_agent(s, "preprocessing"))
    graph.add_node("preprocessing", preprocessing_agent)

    graph.add_node("baseline", baseline_model_agent)
    graph.add_node("decision_model", lambda s: decision_agent(s, "model_selection"))

    graph.add_node("tuning", hyperparameter_tuning_agent)
    graph.add_node("notebook", notebook_agent)

    # Flow
    graph.set_entry_point("profiling")

    graph.add_conditional_edges(
        "profiling",
        lambda state: route_on_error(state, "eda"),
        {
            "eda": "eda",
            END: END,
        }
    )
    graph.add_conditional_edges(
        "eda",
        lambda state: route_on_error(state, "stats"),
        {
            "stats": "stats",
            END: END,
        }
    )

    graph.add_conditional_edges(
        "stats",
        lambda state: route_on_error(state, "decision_pre"),
        {
            "decision_pre": "decision_pre",
            END: END,
        }
    )

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

    graph.add_conditional_edges(
        "preprocessing",
        lambda state: route_on_error(state, "baseline"),
        {
            "baseline": "baseline",
            END: END,
        }
    )
    graph.add_conditional_edges(
        "baseline",
        lambda state: route_on_error(state, "decision_model"),
        {
            "decision_model": "decision_model",
            END: END,
        }
    )

    # Model decision routing
    def route_model(state):
        score = state["baseline_result"].get("score")
        metric = state["baseline_result"].get("metric", "accuracy")

        if metric in ("accuracy", "f1_weighted"):
            strong_enough = score is not None and score > 0.8
        elif metric == "r2":
            strong_enough = score is not None and score > 0.6
        else:
            strong_enough = False

        if strong_enough:
            return "notebook"
        else:
            return "tuning"

    graph.add_conditional_edges(
        "decision_model",
        route_model,
        {
            "tuning": "tuning",
            "notebook": "notebook"
        }
    )

    graph.add_conditional_edges(
        "tuning",
        lambda state: route_on_error(state, "notebook"),
        {
            "notebook": "notebook",
            END: END,
        }
    )
    graph.add_edge("notebook", END)

    return graph.compile()
