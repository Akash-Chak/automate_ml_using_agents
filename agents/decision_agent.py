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


def _fallback_fe_from_profiling(state) -> dict:
    """
    Build a feature_engineering dict from profiling_agent's deterministic
    fe_recommendations when the LLM is unavailable.
    """
    profiling = state.get("profiling_report", {})
    fe_recs = profiling.get("recommendations", {}).get("fe_recommendations", {})
    result = {}
    for col, rec in fe_recs.items():
        transform = rec.get("suggested_transform")
        if transform == "drop":
            action, method = "drop", None
        elif transform in ("log1p", "yeo_johnson", "winsorize"):
            action, method = "transform", transform
        elif transform == "cyclical":
            action, method = "decompose", "cyclical"
        elif transform == "target_encode":
            action, method = "encode", "target_encode"
        else:
            action, method = "keep", None

        if action != "keep":
            result[col] = {
                "action": action,
                "method": method,
                "reason": rec.get("reason", ""),
            }
    return result


def fallback_decision(stage="preprocessing", state=None):
    problem_type = (state or {}).get("problem_type", "classification")
    fe = _fallback_fe_from_profiling(state or {}) if stage == "preprocessing" else {}

    if stage == "preprocessing":
        return {
            "action": "proceed",
            "reason": "Using rule-based fallback because the LLM decision service is unavailable.",
            "feature_engineering": fe,
            "interaction_features": [],
            "drop_features": [],
            "recommended_model": "baseline_first",
            "preprocessing_steps": [],
            "candidate_models": [],
            "tuning_strategy": {},
        }

    return {
        "action": "proceed",
        "reason": "Using rule-based fallback because the LLM decision service is unavailable.",
        "feature_engineering": {},
        "interaction_features": [],
        "drop_features": [],
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
    decision.setdefault("feature_engineering", {})
    decision.setdefault("interaction_features", [])
    decision.setdefault("drop_features", [])
    decision.setdefault("recommended_model",
                        "baseline_first" if stage == "preprocessing" else "hyperparameter_tuning_agent")
    decision.setdefault("preprocessing_steps", [])

    if stage == "model_selection":
        default_strategy = {
            "cv_folds": 3,
            "optuna_trials_per_model": 12,
            "use_class_weight_if_imbalanced": True,
        }
        candidate_models = decision.get("candidate_models")
        if not isinstance(candidate_models, list) or not candidate_models:
            decision["candidate_models"] = _default_tuning_candidates(
                state.get("problem_type", "classification"))
        decision["tuning_strategy"] = {
            **default_strategy,
            **(decision.get("tuning_strategy") or {}),
        }
    else:
        decision.setdefault("candidate_models", [])
        decision.setdefault("tuning_strategy", {})

    return decision


def _build_preprocessing_prompt(context: dict) -> str:
    return f"""You are a senior data scientist performing feature engineering analysis.

You have been given profiling, EDA, and statistical analysis reports for a dataset.
Your job is to decide exactly how to engineer each feature to maximise predictive signal.

## Analysis Reports
{json.dumps(context, indent=2)}

## Your Task

For every non-target feature, specify a feature engineering action.
Also identify any interaction features worth creating (ratios, products, differences).

Available actions per column:
- "transform": apply a mathematical transform (log1p, yeo_johnson, bin_quantile, winsorize, sqrt)
- "encode": apply an encoding strategy (target_encode, one_hot, frequency, ordinal)
- "decompose": extract components from datetime columns (cyclical encoding for month/day-of-week, plus year/month/day numeric)
- "drop": remove the column entirely
- "keep": leave as-is with standard imputation only

Use "yeo_johnson" instead of "log1p" for any column with negative values.
Use "target_encode" for high-cardinality categoricals (cardinality_ratio > 0.2).
Use "bin_quantile" for numeric features flagged as non-linear candidates.
Flag interaction features when two features are both correlated with the target but in different directions,
or when a ratio/difference is more semantically meaningful than the raw values.

Output STRICT JSON only — no markdown, no explanation outside the JSON:
{{
    "action": "proceed",
    "reason": "<one sentence summarising the data and FE strategy>",
    "feature_engineering": {{
        "<column_name>": {{
            "action": "transform | encode | decompose | drop | keep",
            "method": "log1p | yeo_johnson | winsorize | bin_quantile | sqrt | target_encode | one_hot | frequency | cyclical | null",
            "reason": "<one sentence>"
        }}
    }},
    "interaction_features": [
        {{"col_a": "<col>", "col_b": "<col>", "type": "ratio | product | difference", "reason": "<why>"}}
    ],
    "drop_features": ["<col>", "..."],
    "recommended_model": "baseline_first",
    "preprocessing_steps": [],
    "candidate_models": [],
    "tuning_strategy": {{}}
}}
"""


def _build_model_selection_prompt(context: dict, problem_type: str) -> str:
    return f"""You are a senior data scientist choosing models for hyperparameter tuning.

## Pipeline Results So Far
{json.dumps(context, indent=2)}

## Your Task

Based on the baseline result and the dataset characteristics, select the most promising
model families for Optuna-based hyperparameter search.

Consider:
- Dataset size and dimensionality
- Class imbalance (if classification)
- Feature types (lots of categoricals → tree models; mostly numeric + linear signal → linear models)
- Baseline model performance
- Non-linear candidates (if any) → prefer tree/ensemble models

Output STRICT JSON only:
{{
    "action": "proceed",
    "reason": "<one sentence>",
    "feature_engineering": {{}},
    "interaction_features": [],
    "drop_features": [],
    "recommended_model": "hyperparameter_tuning_agent",
    "preprocessing_steps": [],
    "candidate_models": ["<model_id>", "..."],
    "tuning_strategy": {{
        "cv_folds": 3,
        "optuna_trials_per_model": 12,
        "use_class_weight_if_imbalanced": true
    }}
}}

Valid model IDs for {problem_type}:
{json.dumps(_default_tuning_candidates(problem_type), indent=2)}
"""


def _build_context(state: dict, stage: str) -> dict:
    profiling = state.get("profiling_report", {})
    stats     = state.get("stats_report", {})
    eda       = state.get("eda_report", {})
    baseline  = state.get("baseline_result", {})

    if stage == "preprocessing":
        return safe_serialize({
            "problem_type": state.get("problem_type", "classification"),
            "profiling": {
                "num_rows":          profiling.get("num_rows"),
                "num_columns":       profiling.get("num_columns"),
                "data_quality_score": profiling.get("data_quality_score"),
                "warnings":          profiling.get("warnings", [])[:10],
                "recommendations": {
                    "fe_recommendations": profiling.get("recommendations", {}).get("fe_recommendations", {}),
                    "skewed_columns":     profiling.get("recommendations", {}).get("skewed_columns", []),
                    "high_cardinality":   profiling.get("recommendations", {}).get("high_cardinality", []),
                    "correlated_pairs":   profiling.get("recommendations", {}).get("correlated_pairs", [])[:5],
                    "drop_candidates":    profiling.get("recommendations", {}).get("drop_candidates", []),
                },
            },
            "stats": {
                "ranked_features":       stats.get("ranked_features", [])[:15],
                "significant_after_fdr": stats.get("significant_after_fdr", []),
                "large_effect_features": stats.get("large_effect_features", []),
                "fe_signals":            stats.get("fe_signals", {}),
                "mi_scores":             stats.get("mi_scores", {}),
                "warnings":              stats.get("warnings", []),
            },
            "eda": {
                "insights": eda.get("insights", [])[:10],
            },
        })

    # model_selection stage
    return safe_serialize({
        "problem_type": state.get("problem_type", "classification"),
        "baseline": baseline,
        "profiling": {
            "num_rows":    profiling.get("num_rows"),
            "num_columns": profiling.get("num_columns"),
        },
        "stats": {
            "significant_after_fdr":  stats.get("significant_after_fdr", []),
            "large_effect_features":  stats.get("large_effect_features", []),
            "fe_signals":             stats.get("fe_signals", {}),
        },
    })


def decision_agent(state, stage="preprocessing"):

    context = _build_context(state, stage)
    problem_type = state.get("problem_type", "classification")

    if stage == "preprocessing":
        prompt = _build_preprocessing_prompt(context)
    else:
        prompt = _build_model_selection_prompt(context, problem_type)

    log_label = f"decision_{stage}"
    try:
        response = call_llm(prompt, agent=log_label)
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
