# orchestrator/graph.py

def run_pipeline(state):

    from agents.profiling_agent import profiling_agent
    from agents.eda_agent import eda_agent
    from agents.stats_agent import stats_agent
    from agents.preprocessing_agent import preprocessing_agent
    from agents.baseline_model_agent import baseline_model_agent
    from agents.advanced_model_agent import advanced_model_agent

    state = profiling_agent(state)
    state = eda_agent(state)
    state = stats_agent(state)
    state = preprocessing_agent(state)
    state = baseline_model_agent(state)
    state = advanced_model_agent(state)

    return state