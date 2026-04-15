from __future__ import annotations

import math
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.data_utils import ensure_processed_data

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _space_categorical(values):
    return {"type": "categorical", "choices": list(values)}


def _space_int(low, high, step=1, log=False):
    return {"type": "int", "low": low, "high": high, "step": step, "log": log}


def _space_float(low, high, step=None, log=False):
    return {"type": "float", "low": low, "high": high, "step": step, "log": log}


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float, int, np.integer)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return value


def _emit_progress(state, event):
    callback = state.get("_progress_callback")
    if callable(callback):
        try:
            callback(event)
        except Exception:
            pass


def _scaled_model(estimator):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _classification_metric_name(state, y):
    baseline_metric = state.get("baseline_result", {}).get("metric")
    if baseline_metric in {"accuracy", "f1_weighted"}:
        return baseline_metric
    class_counts = y.value_counts()
    imbalance_ratio = float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0
    return "f1_weighted" if imbalance_ratio < 0.35 else "accuracy"


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


def _classification_registry():
    registry = {
        "logistic_regression": {
            "label": "Logistic Regression",
            "builder": lambda class_weight=None: _scaled_model(
                LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight)
            ),
            "grid": {
                "model__C": _space_float(1e-3, 50.0, log=True),
            },
            "supports_class_weight": True,
            "note": "Linear classification baseline with regularization.",
        },
        "ridge_classifier": {
            "label": "Ridge Classifier",
            "builder": lambda class_weight=None: _scaled_model(
                RidgeClassifier(class_weight=class_weight)
            ),
            "grid": {
                "model__alpha": _space_float(1e-3, 20.0, log=True),
            },
            "supports_class_weight": True,
            "note": "Stable linear classifier for wide tabular feature spaces.",
        },
        "linear_svc": {
            "label": "Linear SVC",
            "builder": lambda class_weight=None: _scaled_model(
                LinearSVC(class_weight=class_weight, max_iter=5000)
            ),
            "grid": {
                "model__C": _space_float(1e-3, 20.0, log=True),
            },
            "supports_class_weight": True,
            "note": "Margin-based linear classifier for separable tabular data.",
        },
        "svc_rbf": {
            "label": "SVC (RBF)",
            "builder": lambda class_weight=None: _scaled_model(
                SVC(kernel="rbf", class_weight=class_weight)
            ),
            "grid": {
                "model__C": _space_float(1e-2, 50.0, log=True),
                "model__gamma": _space_categorical(["scale", "auto"]),
            },
            "supports_class_weight": True,
            "note": "Kernel SVM for non-linear decision boundaries.",
        },
        "knn_classifier": {
            "label": "KNN Classifier",
            "builder": lambda class_weight=None: _scaled_model(KNeighborsClassifier()),
            "grid": {
                "model__n_neighbors": _space_int(3, 25, step=2),
                "model__weights": _space_categorical(["uniform", "distance"]),
            },
            "supports_class_weight": False,
            "note": "Distance-based local classifier.",
        },
        "gaussian_nb": {
            "label": "Gaussian NB",
            "builder": lambda class_weight=None: GaussianNB(),
            "grid": {
                "var_smoothing": _space_float(1e-11, 1e-6, log=True),
            },
            "supports_class_weight": False,
            "note": "Fast probabilistic baseline for simple feature distributions.",
        },
        "decision_tree_classifier": {
            "label": "Decision Tree Classifier",
            "builder": lambda class_weight=None: DecisionTreeClassifier(
                random_state=42,
                class_weight=class_weight,
            ),
            "grid": {
                "max_depth": _space_categorical([None, 4, 6, 8, 12, 16]),
                "min_samples_leaf": _space_int(1, 12),
                "min_samples_split": _space_int(2, 20),
            },
            "supports_class_weight": True,
            "note": "Single-tree non-linear classifier.",
        },
        "random_forest_classifier": {
            "label": "Random Forest Classifier",
            "builder": lambda class_weight=None: RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
            ),
            "grid": {
                "n_estimators": _space_int(150, 500, step=50),
                "max_depth": _space_categorical([None, 8, 12, 16, 24]),
                "min_samples_leaf": _space_int(1, 8),
                "max_features": _space_categorical(["sqrt", "log2", None]),
            },
            "supports_class_weight": True,
            "note": "Bagged tree ensemble.",
        },
        "extra_trees_classifier": {
            "label": "Extra Trees Classifier",
            "builder": lambda class_weight=None: ExtraTreesClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
            ),
            "grid": {
                "n_estimators": _space_int(150, 500, step=50),
                "max_depth": _space_categorical([None, 8, 12, 16, 24]),
                "min_samples_leaf": _space_int(1, 6),
                "max_features": _space_categorical(["sqrt", "log2", None]),
            },
            "supports_class_weight": True,
            "note": "Highly randomized bagging-style ensemble.",
        },
        "gradient_boosting_classifier": {
            "label": "Gradient Boosting Classifier",
            "builder": lambda class_weight=None: GradientBoostingClassifier(random_state=42),
            "grid": {
                "n_estimators": _space_int(50, 300, step=25),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "max_depth": _space_int(2, 5),
                "subsample": _space_float(0.6, 1.0),
            },
            "supports_class_weight": False,
            "note": "Classic boosting ensemble for non-linear tabular data.",
        },
        "hist_gradient_boosting_classifier": {
            "label": "Hist Gradient Boosting Classifier",
            "builder": lambda class_weight=None: HistGradientBoostingClassifier(random_state=42),
            "grid": {
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "max_depth": _space_categorical([None, 4, 6, 8, 12]),
                "max_leaf_nodes": _space_int(15, 127, step=8),
                "min_samples_leaf": _space_int(10, 50, step=5),
            },
            "supports_class_weight": False,
            "note": "Histogram-based boosting for efficient tabular learning.",
        },
        "ada_boost_classifier": {
            "label": "AdaBoost Classifier",
            "builder": lambda class_weight=None: AdaBoostClassifier(random_state=42),
            "grid": {
                "n_estimators": _space_int(25, 250, step=25),
                "learning_rate": _space_float(0.01, 2.0, log=True),
            },
            "supports_class_weight": False,
            "note": "Adaptive boosting ensemble.",
        },
        "bagging_classifier": {
            "label": "Bagging Classifier",
            "builder": lambda class_weight=None: BaggingClassifier(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": _space_int(10, 100, step=10),
                "max_samples": _space_float(0.5, 1.0),
                "max_features": _space_float(0.5, 1.0),
            },
            "supports_class_weight": False,
            "note": "Bootstrap aggregation over base learners.",
        },
        "mlp_classifier": {
            "label": "MLP Classifier",
            "builder": lambda class_weight=None: _scaled_model(
                MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
            ),
            "grid": {
                "model__hidden_layer_sizes": _space_categorical([(64,), (128,), (256,), (64, 32), (128, 64)]),
                "model__alpha": _space_float(1e-6, 1e-1, log=True),
                "model__learning_rate_init": _space_float(1e-4, 5e-2, log=True),
            },
            "supports_class_weight": False,
            "note": "Feed-forward neural baseline for dense processed features.",
        },
    }

    if XGBClassifier is not None:
        registry["xgboost_classifier"] = {
            "label": "XGBoost Classifier",
            "builder": lambda class_weight=None: XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_estimators=200,
                n_jobs=-1,
            ),
            "grid": {
                "max_depth": _space_int(3, 10),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "subsample": _space_float(0.6, 1.0),
                "colsample_bytree": _space_float(0.6, 1.0),
            },
            "supports_class_weight": False,
            "note": "External gradient boosting model when xgboost is available.",
        }

    if LGBMClassifier is not None:
        registry["lightgbm_classifier"] = {
            "label": "LightGBM Classifier",
            "builder": lambda class_weight=None: LGBMClassifier(
                random_state=42,
                n_estimators=200,
                class_weight=class_weight,
            ),
            "grid": {
                "num_leaves": _space_int(15, 127, step=4),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "subsample": _space_float(0.6, 1.0),
                "min_child_samples": _space_int(5, 50),
            },
            "supports_class_weight": True,
            "note": "External gradient boosting model when lightgbm is available.",
        }

    return registry


def _regression_registry():
    registry = {
        "linear_regression": {
            "label": "Linear Regression",
            "builder": lambda: _scaled_model(LinearRegression()),
            "grid": {
                "model__fit_intercept": _space_categorical([True, False]),
            },
            "note": "Ordinary least squares baseline.",
        },
        "ridge_regressor": {
            "label": "Ridge Regressor",
            "builder": lambda: _scaled_model(Ridge(random_state=42)),
            "grid": {
                "model__alpha": _space_float(1e-4, 50.0, log=True),
            },
            "note": "L2-regularized linear regressor.",
        },
        "lasso_regressor": {
            "label": "Lasso Regressor",
            "builder": lambda: _scaled_model(Lasso(random_state=42, max_iter=5000)),
            "grid": {
                "model__alpha": _space_float(1e-6, 1.0, log=True),
            },
            "note": "Sparse linear regressor.",
        },
        "elasticnet_regressor": {
            "label": "ElasticNet Regressor",
            "builder": lambda: _scaled_model(ElasticNet(random_state=42, max_iter=5000)),
            "grid": {
                "model__alpha": _space_float(1e-6, 1.0, log=True),
                "model__l1_ratio": _space_float(0.05, 0.95),
            },
            "note": "Mixed L1/L2 regularized regressor.",
        },
        "linear_svr": {
            "label": "Linear SVR",
            "builder": lambda: _scaled_model(LinearSVR(random_state=42, max_iter=5000)),
            "grid": {
                "model__C": _space_float(1e-3, 20.0, log=True),
                "model__epsilon": _space_float(1e-4, 1.0, log=True),
            },
            "note": "Linear support vector regressor.",
        },
        "svr_rbf": {
            "label": "SVR (RBF)",
            "builder": lambda: _scaled_model(SVR(kernel="rbf")),
            "grid": {
                "model__C": _space_float(1e-2, 50.0, log=True),
                "model__epsilon": _space_float(1e-4, 1.0, log=True),
                "model__gamma": _space_categorical(["scale", "auto"]),
            },
            "note": "Kernel support vector regressor.",
        },
        "knn_regressor": {
            "label": "KNN Regressor",
            "builder": lambda: _scaled_model(KNeighborsRegressor()),
            "grid": {
                "model__n_neighbors": _space_int(3, 25, step=2),
                "model__weights": _space_categorical(["uniform", "distance"]),
            },
            "note": "Distance-based local regressor.",
        },
        "decision_tree_regressor": {
            "label": "Decision Tree Regressor",
            "builder": lambda: DecisionTreeRegressor(random_state=42),
            "grid": {
                "max_depth": _space_categorical([None, 4, 6, 8, 12, 16]),
                "min_samples_leaf": _space_int(1, 12),
                "min_samples_split": _space_int(2, 20),
            },
            "note": "Single-tree non-linear regressor.",
        },
        "random_forest_regressor": {
            "label": "Random Forest Regressor",
            "builder": lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": _space_int(150, 500, step=50),
                "max_depth": _space_categorical([None, 8, 12, 16, 24]),
                "min_samples_leaf": _space_int(1, 8),
                "max_features": _space_categorical(["sqrt", "log2", None]),
            },
            "note": "Bagged tree ensemble for regression.",
        },
        "extra_trees_regressor": {
            "label": "Extra Trees Regressor",
            "builder": lambda: ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": _space_int(150, 500, step=50),
                "max_depth": _space_categorical([None, 8, 12, 16, 24]),
                "min_samples_leaf": _space_int(1, 6),
                "max_features": _space_categorical(["sqrt", "log2", None]),
            },
            "note": "Highly randomized tree ensemble for regression.",
        },
        "gradient_boosting_regressor": {
            "label": "Gradient Boosting Regressor",
            "builder": lambda: GradientBoostingRegressor(random_state=42),
            "grid": {
                "n_estimators": _space_int(50, 300, step=25),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "max_depth": _space_int(2, 5),
                "subsample": _space_float(0.6, 1.0),
            },
            "note": "Classic boosting regressor.",
        },
        "hist_gradient_boosting_regressor": {
            "label": "Hist Gradient Boosting Regressor",
            "builder": lambda: HistGradientBoostingRegressor(random_state=42),
            "grid": {
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "max_depth": _space_categorical([None, 4, 6, 8, 12]),
                "max_leaf_nodes": _space_int(15, 127, step=8),
                "min_samples_leaf": _space_int(10, 50, step=5),
            },
            "note": "Histogram-based boosting regressor.",
        },
        "ada_boost_regressor": {
            "label": "AdaBoost Regressor",
            "builder": lambda: AdaBoostRegressor(random_state=42),
            "grid": {
                "n_estimators": _space_int(25, 250, step=25),
                "learning_rate": _space_float(0.01, 2.0, log=True),
            },
            "note": "Adaptive boosting regressor.",
        },
        "bagging_regressor": {
            "label": "Bagging Regressor",
            "builder": lambda: BaggingRegressor(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": _space_int(10, 100, step=10),
                "max_samples": _space_float(0.5, 1.0),
                "max_features": _space_float(0.5, 1.0),
            },
            "note": "Bootstrap aggregation for regression.",
        },
        "mlp_regressor": {
            "label": "MLP Regressor",
            "builder": lambda: _scaled_model(
                MLPRegressor(random_state=42, max_iter=500, early_stopping=True)
            ),
            "grid": {
                "model__hidden_layer_sizes": _space_categorical([(64,), (128,), (256,), (64, 32), (128, 64)]),
                "model__alpha": _space_float(1e-6, 1e-1, log=True),
                "model__learning_rate_init": _space_float(1e-4, 5e-2, log=True),
            },
            "note": "Feed-forward neural regressor.",
        },
    }

    if XGBRegressor is not None:
        registry["xgboost_regressor"] = {
            "label": "XGBoost Regressor",
            "builder": lambda: XGBRegressor(
                random_state=42,
                n_estimators=200,
                n_jobs=-1,
            ),
            "grid": {
                "max_depth": _space_int(3, 10),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "subsample": _space_float(0.6, 1.0),
                "colsample_bytree": _space_float(0.6, 1.0),
            },
            "note": "External boosting regressor when xgboost is available.",
        }

    if LGBMRegressor is not None:
        registry["lightgbm_regressor"] = {
            "label": "LightGBM Regressor",
            "builder": lambda: LGBMRegressor(
                random_state=42,
                n_estimators=200,
            ),
            "grid": {
                "num_leaves": _space_int(15, 127, step=4),
                "learning_rate": _space_float(0.01, 0.3, log=True),
                "subsample": _space_float(0.6, 1.0),
                "min_child_samples": _space_int(5, 50),
            },
            "note": "External boosting regressor when lightgbm is available.",
        }

    return registry


def _suggest_params(trial, search_space):
    params = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict):
            space_type = spec.get("type")
            if space_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif space_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
            elif space_type == "float":
                float_kwargs = {
                    "step": spec.get("step"),
                    "log": spec.get("log", False),
                }
                if float_kwargs["step"] is None:
                    float_kwargs.pop("step")
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    **float_kwargs,
                )
            else:
                params[name] = trial.suggest_categorical(name, spec.get("choices", [spec]))
        else:
            params[name] = trial.suggest_categorical(name, spec)
    return params


def _classification_strategies(spec, use_balancing):
    if use_balancing and spec.get("supports_class_weight"):
        return [("unweighted", None), ("balanced", "balanced")]
    return [("default", None)]


def _trial_record(base_record, scores, params):
    return {
        **base_record,
        "params": params,
        "cv_score": round(scores["cv_score"], 4),
        "holdout_primary_score": round(scores["primary_score"], 4),
        "accuracy": _safe_float(round(scores.get("accuracy", 0.0), 4)) if "accuracy" in scores else None,
        "f1_weighted": _safe_float(round(scores.get("f1_weighted", 0.0), 4)) if "f1_weighted" in scores else None,
        "r2": _safe_float(round(scores.get("r2", 0.0), 4)) if "r2" in scores else None,
        "mae": _safe_float(round(scores.get("mae", 0.0), 4)) if "mae" in scores else None,
        "rmse": _safe_float(round(scores.get("rmse", 0.0), 4)) if "rmse" in scores else None,
    }


def hyperparameter_tuning_agent(state):
    target = state["target_column"]
    problem_type = state["problem_type"]
    decision = state.get("decision_log", {}).get("model_selection", {})
    tuning_strategy = decision.get("tuning_strategy", {})
    requested_models = decision.get("candidate_models", [])

    tuning_mode = state.get("tuning_mode", "smoke_test")

    df = ensure_processed_data(state)
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    cv_folds = int(tuning_strategy.get("cv_folds", 3) or 3)
    optuna_trials_per_model = int(
        tuning_strategy.get(
            "optuna_trials_per_model",
            tuning_strategy.get("max_trials_per_model", 12),
        ) or 12
    )
    use_class_weight_if_imbalanced = bool(tuning_strategy.get("use_class_weight_if_imbalanced", True))

    if problem_type == "classification":
        registry = _classification_registry()
        primary_metric = _classification_metric_name(state, y)
        scoring = primary_metric
        class_counts = y.value_counts()
        imbalance_ratio = float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0
        use_balancing = use_class_weight_if_imbalanced and imbalance_ratio < 0.35
    else:
        registry = _regression_registry()
        primary_metric = "r2"
        scoring = "r2"
        imbalance_ratio = None
        use_balancing = False

    model_ids = [model_id for model_id in requested_models if model_id in registry]
    if not model_ids:
        model_ids = list(registry.keys())

    # ── Tuning mode: smoke_test (fast, 1 trial) or full_search (Optuna TPE) ──
    if tuning_mode == "smoke_test":
        optuna_trials_per_model = 1
        sampler_factory = lambda: optuna.samplers.RandomSampler(seed=42)
    else:
        # full_search: use decision agent's trial count and TPE sampler
        sampler_factory = lambda: optuna.samplers.TPESampler(seed=42)

    tuning_history = []
    candidate_results = []
    best_result = None
    best_estimator = None

    _emit_progress(
        state,
        {
            "phase": "start",
            "problem_type": problem_type,
            "metric": primary_metric,
            "models": model_ids,
            "cv_folds": cv_folds,
            "optuna_trials_per_model": optuna_trials_per_model,
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        for model_index, model_id in enumerate(model_ids, start=1):
            spec = registry[model_id]
            strategies = _classification_strategies(spec, use_balancing) if problem_type == "classification" else [("default", None)]

            _emit_progress(
                state,
                {
                    "phase": "start_model",
                    "model_id": model_id,
                    "label": spec["label"],
                    "model_index": model_index,
                    "model_count": len(model_ids),
                    "strategies": [name for name, _ in strategies],
                },
            )

            model_best = None

            for strategy_name, class_weight in strategies:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=sampler_factory(),
                    study_name=f"{model_id}_{strategy_name}",
                )

                def objective(trial):
                    params = _suggest_params(trial, spec["grid"])
                    estimator = spec["builder"](class_weight) if problem_type == "classification" else spec["builder"]()
                    estimator.set_params(**params)

                    cv_scores = cross_val_score(
                        estimator,
                        X_train,
                        y_train,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=1,
                    )
                    estimator = clone(estimator)
                    if problem_type == "classification":
                        holdout_scores = _score_classification(estimator, X_train, X_test, y_train, y_test)
                    else:
                        holdout_scores = _score_regression(estimator, X_train, X_test, y_train, y_test)

                    scores = {
                        "cv_score": float(np.mean(cv_scores)),
                        "primary_score": float(holdout_scores[primary_metric]),
                        **holdout_scores,
                    }
                    trial.set_user_attr("scores", scores)
                    trial.set_user_attr("params", params)
                    return scores["cv_score"]

                for trial_index in range(1, optuna_trials_per_model + 1):
                    try:
                        study.optimize(objective, n_trials=1, catch=(Exception,))
                        trial = study.trials[-1]
                    except Exception as exc:
                        event = {
                            "phase": "trial_error",
                            "model_id": model_id,
                            "label": spec["label"],
                            "strategy": strategy_name,
                            "params": {},
                            "error": str(exc),
                        }
                        tuning_history.append(event)
                        _emit_progress(state, event)
                        continue

                    if trial.state != optuna.trial.TrialState.COMPLETE:
                        event = {
                            "phase": "trial_error",
                            "model_id": model_id,
                            "label": spec["label"],
                            "strategy": strategy_name,
                            "params": dict(trial.params),
                            "error": "Optuna trial did not complete successfully.",
                        }
                        tuning_history.append(event)
                        _emit_progress(state, event)
                        continue

                    params = dict(trial.params)
                    scores = dict(trial.user_attrs.get("scores", {}))
                    record = _trial_record(
                        {
                            "model_id": model_id,
                            "model_label": spec["label"],
                            "strategy": strategy_name,
                            "trial_index": trial_index,
                            "note": spec["note"],
                            "optuna_trial_number": trial.number,
                        },
                        scores,
                        params,
                    )
                    tuning_history.append(record)
                    _emit_progress(
                        state,
                        {
                            "phase": "trial_complete",
                            "model_id": model_id,
                            "label": spec["label"],
                            "strategy": strategy_name,
                            "trial_index": trial_index,
                            "trial_count": optuna_trials_per_model,
                            "optuna_trial_number": trial.number,
                            "params": params,
                            "metric": primary_metric,
                            "cv_score": round(scores["cv_score"], 4),
                            "holdout_primary_score": round(scores["primary_score"], 4),
                        },
                    )

                    if model_best is None or scores["cv_score"] > model_best["cv_score"]:
                        model_best = {
                            "model_id": model_id,
                            "model_label": spec["label"],
                            "strategy": strategy_name,
                            "params": params,
                            "cv_score": scores["cv_score"],
                            "note": spec["note"],
                            **{
                                metric_name: metric_value
                                for metric_name, metric_value in scores.items()
                                if metric_name not in {"cv_score", "primary_score"}
                            },
                        }
                        best_estimator_candidate = spec["builder"](class_weight) if problem_type == "classification" else spec["builder"]()
                        best_estimator_candidate.set_params(**params)
                        if best_result is None or scores["cv_score"] > best_result["cv_score"]:
                            best_result = {
                                **model_best,
                                "metric": primary_metric,
                            }
                            best_estimator = best_estimator_candidate
                            _emit_progress(
                                state,
                                {
                                    "phase": "best_update",
                                    "model_id": model_id,
                                    "label": spec["label"],
                                    "strategy": strategy_name,
                                    "params": params,
                                    "metric": primary_metric,
                                    "cv_score": round(scores["cv_score"], 4),
                                    "holdout_primary_score": round(scores["primary_score"], 4),
                                },
                            )

            if model_best is not None:
                candidate_results.append(
                    {
                        "model_id": model_best["model_id"],
                        "model": model_best["model_label"],
                        "strategy": model_best["strategy"],
                        "cv_score": round(model_best["cv_score"], 4),
                        "metric": primary_metric,
                        "score": round(model_best[primary_metric], 4),
                        "accuracy": _safe_float(round(model_best.get("accuracy", 0.0), 4)) if "accuracy" in model_best else None,
                        "f1_weighted": _safe_float(round(model_best.get("f1_weighted", 0.0), 4)) if "f1_weighted" in model_best else None,
                        "r2": _safe_float(round(model_best.get("r2", 0.0), 4)) if "r2" in model_best else None,
                        "mae": _safe_float(round(model_best.get("mae", 0.0), 4)) if "mae" in model_best else None,
                        "rmse": _safe_float(round(model_best.get("rmse", 0.0), 4)) if "rmse" in model_best else None,
                        "best_params": model_best["params"],
                        "note": model_best["note"],
                    }
                )

    if best_result is None or best_estimator is None:
        return {**state, "error": "Hyperparameter tuning failed for all candidate models."}

    if problem_type == "classification":
        final_scores = _score_classification(best_estimator, X_train, X_test, y_train, y_test)
    else:
        final_scores = _score_regression(best_estimator, X_train, X_test, y_train, y_test)

    _emit_progress(
        state,
        {
            "phase": "completed",
            "model_id": best_result["model_id"],
            "label": best_result["model_label"],
            "strategy": best_result["strategy"],
            "params": best_result["params"],
            "metric": primary_metric,
            "score": round(final_scores[primary_metric], 4),
        },
    )

    advanced_result = {
        "model": best_result["model_label"],
        "model_id": best_result["model_id"],
        "score": _safe_float(final_scores[primary_metric]),
        "metric": primary_metric,
        "score_direction": "higher_is_better",
        "selection_reason": decision.get(
            "reason",
            "Selected using the decision agent candidate list and cross-validated hyperparameter tuning.",
        ),
        "model_reason": best_result["note"],
        "best_hyperparameters": best_result["params"],
        "tuning_cv_score": _safe_float(best_result["cv_score"]),
        "tuning_strategy_selected": best_result["strategy"],
        "candidate_models_considered": model_ids,
        "candidate_results": candidate_results,
        "tuning_history": tuning_history,
        "tuning_summary": {
            "cv_folds": cv_folds,
            "optimizer": "optuna_tpe",
            "optuna_trials_per_model": optuna_trials_per_model,
            "models_requested_by_decision_agent": requested_models,
            "models_evaluated": model_ids,
        },
    }

    if imbalance_ratio is not None:
        advanced_result["class_balance_ratio"] = round(imbalance_ratio, 4)

    if problem_type == "classification":
        advanced_result["accuracy"] = _safe_float(final_scores["accuracy"])
        advanced_result["f1_weighted"] = _safe_float(final_scores["f1_weighted"])
    else:
        advanced_result["r2"] = _safe_float(final_scores["r2"])
        advanced_result["mae"] = _safe_float(final_scores["mae"])
        advanced_result["rmse"] = _safe_float(final_scores["rmse"])

    state["advanced_result"] = advanced_result
    return state
