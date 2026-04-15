from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.data_utils import ensure_processed_data


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float, int, np.integer)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return value


def _classification_context(state, X: pd.DataFrame, y: pd.Series):
    preprocessing = state.get("preprocessing_report", {})
    stats = state.get("stats_report", {})
    profiling = state.get("profiling_report", {})

    class_counts = y.value_counts()
    imbalance_ratio = float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0
    high_dimensional = X.shape[1] > max(30, int(X.shape[0] * 0.35))
    many_sparse_features = preprocessing.get("cat_features", 0) > preprocessing.get("num_features", 0)
    significant_features = len(stats.get("significant_after_fdr", []))
    low_quality = profiling.get("data_quality_score", {}).get("overall_score", 100) < 70

    return {
        "imbalance_ratio": imbalance_ratio,
        "n_classes": int(y.nunique()),
        "high_dimensional": high_dimensional,
        "many_sparse_features": many_sparse_features,
        "significant_features": significant_features,
        "low_quality": low_quality,
    }


def _regression_context(state, X: pd.DataFrame):
    preprocessing = state.get("preprocessing_report", {})
    stats = state.get("stats_report", {})
    profiling = state.get("profiling_report", {})

    transformed_columns = preprocessing.get("transformed_columns", [])
    strong_features = len(stats.get("large_effect_features", [])) + len(stats.get("medium_effect_features", []))
    high_dimensional = X.shape[1] > max(25, int(X.shape[0] * 0.25))
    low_quality = profiling.get("data_quality_score", {}).get("overall_score", 100) < 70

    return {
        "transformed_columns": transformed_columns,
        "strong_features": strong_features,
        "high_dimensional": high_dimensional,
        "low_quality": low_quality,
    }


def _classification_candidates(context, class_weight=None):
    logistic_solver = "liblinear" if context["n_classes"] <= 2 else "lbfgs"
    candidates = []

    if class_weight is None:
        candidates.append(
            (
                "DummyClassifier",
                DummyClassifier(strategy="most_frequent"),
                "Sanity-check majority baseline.",
            )
        )

    candidates.extend([
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1500, solver=logistic_solver, class_weight=class_weight),
            "Strong default for encoded tabular data.",
        ),
        (
            "RidgeClassifier",
            RidgeClassifier(class_weight=class_weight),
            "Stable linear baseline for many sparse features.",
        ),
    ])

    if not context["high_dimensional"] or context["significant_features"] < 8:
        candidates.append(
            (
                "DecisionTreeClassifier",
                DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight=class_weight, random_state=42),
                "Captures simple non-linear interactions with a small tree.",
            )
        )

    return candidates


def _format_classification_model_name(model_name, class_weight):
    if class_weight == "balanced":
        return f"{model_name} (class_weight=balanced)"
    return model_name


def _build_imbalance_selection_reason(primary_metric, tested_imbalance_handling, chosen_strategy):
    metric_reason = (
        "Used weighted F1 as the primary metric because the target is imbalanced."
        if primary_metric == "f1_weighted"
        else "Used accuracy as the primary metric because class balance is acceptable."
    )

    if not tested_imbalance_handling:
        return metric_reason

    strategy_reason = (
        "Comparing class imbalance handling improved the baseline result, so the balanced variant was selected."
        if chosen_strategy == "balanced"
        else "Comparing class imbalance handling did not improve the baseline result, so the unweighted variant was kept."
    )
    return f"{metric_reason} {strategy_reason}"


def _regression_candidates(context):
    candidates = [
        (
            "DummyRegressor",
            DummyRegressor(strategy="mean"),
            "Sanity-check average prediction baseline.",
        ),
        (
            "LinearRegression",
            LinearRegression(),
            "Simple linear baseline after preprocessing and scaling.",
        ),
        (
            "Ridge",
            Ridge(alpha=1.0, random_state=42),
            "Regularized linear baseline for correlated features.",
        ),
        (
            "Lasso",
            Lasso(alpha=0.001, random_state=42, max_iter=5000),
            "Sparse linear baseline that can zero weak coefficients.",
        ),
    ]

    if not context["high_dimensional"]:
        candidates.append(
            (
                "DecisionTreeRegressor",
                DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42),
                "Captures simple non-linear patterns for tabular regression.",
            )
        )

    return candidates


def _score_classification(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
    }


def _score_regression(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
    }


def baseline_model_agent(state):
    target = state["target_column"]
    problem_type = state["problem_type"]

    df = ensure_processed_data(state)
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    candidate_results = []

    if problem_type == "classification":
        context = _classification_context(state, X, y)
        primary_metric = "f1_weighted" if context["imbalance_ratio"] < 0.35 else "accuracy"
        should_test_imbalance_handling = context["imbalance_ratio"] < 0.35

        strategy_candidates = [("unweighted", _classification_candidates(context, class_weight=None))]
        if should_test_imbalance_handling:
            strategy_candidates.append(("balanced", _classification_candidates(context, class_weight="balanced")))

        best_model = None
        best_scores = None
        best_name = None
        best_note = None
        best_strategy = "unweighted"
        strategy_results = {}

        for strategy_name, candidates in strategy_candidates:
            strategy_best = None
            strategy_best_name = None
            strategy_best_note = None

            for model_name, model, note in candidates:
                scores = _score_classification(model, X_train, X_test, y_train, y_test)
                display_name = _format_classification_model_name(
                    model_name,
                    "balanced" if strategy_name == "balanced" else None,
                )
                row = {
                    "model": display_name,
                    "accuracy": round(scores["accuracy"], 4),
                    "f1_weighted": round(scores["f1_weighted"], 4),
                    "imbalance_strategy": strategy_name,
                    "note": note,
                }
                candidate_results.append(row)

                if strategy_best is None or scores[primary_metric] > strategy_best[primary_metric]:
                    strategy_best = scores
                    strategy_best_name = display_name
                    strategy_best_note = note

                if best_scores is None or scores[primary_metric] > best_scores[primary_metric]:
                    best_model = model
                    best_scores = scores
                    best_name = display_name
                    best_note = note
                    best_strategy = strategy_name

            strategy_results[strategy_name] = {
                "model": strategy_best_name,
                "score": float(strategy_best[primary_metric]),
                "accuracy": float(strategy_best["accuracy"]),
                "f1_weighted": float(strategy_best["f1_weighted"]),
                "metric": primary_metric,
                "selected": strategy_name == best_strategy,
                "note": strategy_best_note,
            }

        selection_reason = _build_imbalance_selection_reason(
            primary_metric=primary_metric,
            tested_imbalance_handling=should_test_imbalance_handling,
            chosen_strategy=best_strategy,
        )

        best_model.fit(X_train, y_train)
        imbalance_comparison = {
            "tested": should_test_imbalance_handling,
            "selected_strategy": best_strategy,
            "metric": primary_metric,
            "without_imbalance_handling": strategy_results.get("unweighted"),
        }
        if should_test_imbalance_handling:
            balanced_result = strategy_results.get("balanced")
            unweighted_result = strategy_results.get("unweighted")
            improvement = None
            if balanced_result and unweighted_result:
                improvement = round(
                    balanced_result["score"] - unweighted_result["score"],
                    4,
                )
            imbalance_comparison["with_imbalance_handling"] = balanced_result
            imbalance_comparison["score_delta_balanced_minus_unweighted"] = improvement

        state["baseline_result"] = {
            "model": best_name,
            "score": float(best_scores[primary_metric]),
            "metric": primary_metric,
            "accuracy": float(best_scores["accuracy"]),
            "f1_weighted": float(best_scores["f1_weighted"]),
            "score_direction": "higher_is_better",
            "selection_reason": selection_reason,
            "model_reason": best_note,
            "class_balance_ratio": round(context["imbalance_ratio"], 4),
            "imbalance_strategy_selected": best_strategy,
            "imbalance_comparison": imbalance_comparison,
            "candidate_results": candidate_results,
        }
        return state

    context = _regression_context(state, X)
    candidates = _regression_candidates(context)
    primary_metric = "r2"
    selection_reason = "Used R² as the primary baseline metric so downstream routing can compare models consistently."

    best_model = None
    best_scores = None
    best_name = None
    best_note = None

    for model_name, model, note in candidates:
        scores = _score_regression(model, X_train, X_test, y_train, y_test)
        row = {
            "model": model_name,
            "r2": round(scores["r2"], 4),
            "mae": round(scores["mae"], 4),
            "rmse": round(scores["rmse"], 4),
            "note": note,
        }
        candidate_results.append(row)

        if best_scores is None or scores[primary_metric] > best_scores[primary_metric]:
            best_model = model
            best_scores = scores
            best_name = model_name
            best_note = note

    best_model.fit(X_train, y_train)
    state["baseline_result"] = {
        "model": best_name,
        "score": _safe_float(best_scores["r2"]),
        "metric": "r2",
        "r2": _safe_float(best_scores["r2"]),
        "mae": _safe_float(best_scores["mae"]),
        "rmse": _safe_float(best_scores["rmse"]),
        "score_direction": "higher_is_better",
        "selection_reason": selection_reason,
        "model_reason": best_note,
        "candidate_results": candidate_results,
    }
    return state
