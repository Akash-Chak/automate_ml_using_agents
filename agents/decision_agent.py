# agents/decision_agent.py

from config import call_llm

import json
import numpy as np
import pandas as pd

def safe_serialize(obj):
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def decision_agent(state, stage="preprocessing"):

    context = {
        "profiling": state.get("profiling_report"),
        "eda": state.get("eda_report"),
        "stats": state.get("stats_report"),
        "baseline": state.get("baseline_result"),
    }

    safe_context = safe_serialize(context)

    prompt = f"""
    You are an expert data scientist.

    Stage: {stage}

    Context:
    {json.dumps(safe_context, indent=2)}

    Tasks:
    - Decide best action for this stage
    - Suggest:
        * preprocessing steps
        * feature selection
        * model choice
        * whether to proceed or retry

    Output STRICT JSON:
    {{
        "action": "...",
        "reason": "...",
        "recommended_model": "...",
        "preprocessing_steps": []
    }}
    """

    response = call_llm(prompt)

    try:
        decision = json.loads(response)
    except:
        decision = {"action": "proceed"}

    state["decision_log"] = state.get("decision_log", {})
    state["decision_log"][stage] = decision

    return state