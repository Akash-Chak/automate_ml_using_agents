from agents.hyperparameter_tuning_agent import hyperparameter_tuning_agent


def advanced_model_agent(state):
    state["advanced_agent_strategy"] = "delegated_to_hyperparameter_tuning_agent"
    return hyperparameter_tuning_agent(state)
