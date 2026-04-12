# agents/decision_agent.py

from config import call_llm

import json
import re
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


def _default_tuning_candidates(problem_type):
    if problem_type == "classification":
        return [
            "logistic_regression",
            "ridge_classifier",
            "linear_svc",
            "svc_rbf",
            "knn_classifier",
            "gaussian_nb",
            "decision_tree_classifier",
            "random_forest_classifier",
            "extra_trees_classifier",
            "gradient_boosting_classifier",
            "hist_gradient_boosting_classifier",
            "ada_boost_classifier",
            "bagging_classifier",
            "mlp_classifier",
        ]

    return [
        "linear_regression",
        "ridge_regressor",
        "lasso_regressor",
        "elasticnet_regressor",
        "linear_svr",
        "svr_rbf",
        "knn_regressor",
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
        "hist_gradient_boosting_regressor",
        "ada_boost_regressor",
        "bagging_regressor",
        "mlp_regressor",
    ]


def fallback_decision(stage="preprocessing", state=None):
    problem_type = (state or {}).get("problem_type", "classification")
    if stage == "preprocessing":
        return {
            "action": "proceed",
            "reason": "Using rule-based fallback because the LLM decision service is unavailable.",
            "recommended_model": "baseline_first",
            "preprocessing_steps": ["impute_missing_values", "one_hot_encode_categoricals"],
            "candidate_models": [],
            "tuning_strategy": {},
        }

    return {
        "action": "proceed",
        "reason": "Using rule-based fallback because the LLM decision service is unavailable.",
        "recommended_model": "hyperparameter_tuning_agent",
        "preprocessing_steps": [],
        "candidate_models": _default_tuning_candidates(problem_type),
        "tuning_strategy": {
            "cv_folds": 3,
            "optuna_trials_per_model": 12,
            "use_class_weight_if_imbalanced": True,
        },
    }


def parse_llm_json(response_text: str):
    text = (response_text or "").strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def normalize_decision(decision, stage, state):
    decision = decision or {}
    decision.setdefault("action", "proceed")
    decision.setdefault("reason", "No reason provided.")
    decision.setdefault("recommended_model", "baseline_first" if stage == "preprocessing" else "hyperparameter_tuning_agent")
    decision.setdefault("preprocessing_steps", [])

    if stage == "model_selection":
        default_strategy = {
            "cv_folds": 3,
            "optuna_trials_per_model": 12,
            "use_class_weight_if_imbalanced": True,
        }
        candidate_models = decision.get("candidate_models")
        if not isinstance(candidate_models, list) or not candidate_models:
            decision["candidate_models"] = _default_tuning_candidates(state.get("problem_type", "classification"))
        decision["tuning_strategy"] = {
            **default_strategy,
            **(decision.get("tuning_strategy") or {}),
        }
    else:
        decision.setdefault("candidate_models", [])
        decision.setdefault("tuning_strategy", {})

    return decision

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
        * candidate model families for tuning if this is model_selection
        * a tuning strategy with cv_folds, optuna_trials_per_model, use_class_weight_if_imbalanced
        * whether to proceed or retry

    Output STRICT JSON:
    {{
        "action": "...",
        "reason": "...",
        "recommended_model": "...",
        "preprocessing_steps": [],
        "candidate_models": [],
        "tuning_strategy": {{}}
    }}
    """

    try:
        response = call_llm(prompt)
        decision = parse_llm_json(response)
        decision = normalize_decision(decision, stage, state)
        decision["llm_status"] = "success"
    except Exception as exc:
        decision = fallback_decision(stage, state)
        decision["llm_status"] = "fallback"
        decision["llm_error"] = str(exc)

    state["decision_log"] = state.get("decision_log", {})
    state["decision_log"][stage] = decision

    return state
