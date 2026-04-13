# utils/mlflow_utils.py
#
# Centralised MLflow helpers. All agents import from here.
# The entire module is a no-op when mlflow is not installed.

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import mlflow
    import mlflow.tracking
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Experiment setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow_experiment(tracking_uri: str, experiment_name: str) -> Optional[str]:
    """Set tracking URI and ensure experiment exists. Returns experiment_id or None."""
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else None
    except Exception:
        return None


def get_experiment_url(tracking_uri: str, experiment_name: str) -> Optional[str]:
    """
    Returns a localhost MLflow UI URL only for local tracking URIs.
    Users need to run `mlflow ui --backend-store-uri <tracking_uri>` separately.
    """
    if not MLFLOW_AVAILABLE:
        return None
    # Only construct URL for local paths (not remote servers)
    is_local = tracking_uri.startswith("./") or tracking_uri.startswith("mlruns") or tracking_uri == "."
    if not is_local:
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        return f"http://localhost:5000/#/experiments/{exp.experiment_id}"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Baseline logging
# ─────────────────────────────────────────────────────────────────────────────

def log_baseline_run(experiment_name: str, state: Dict, baseline_result: Dict) -> Optional[str]:
    """
    Opens a parent MLflow run for the baseline phase and logs each candidate
    model as a nested child run. Returns the parent run_id or None.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        candidate_results = baseline_result.get("candidate_results", [])
        problem_type      = state.get("problem_type", "")
        dataset_path      = state.get("dataset_path", "")
        target_column     = state.get("target_column", "")

        with mlflow.start_run(run_name="baseline", tags={
            "run_type":     "baseline",
            "target_column": target_column,
            "problem_type": problem_type,
        }) as parent_run:
            mlflow.log_params({
                "dataset_path":  dataset_path,
                "target_column": target_column,
                "problem_type":  problem_type,
                "n_candidates":  len(candidate_results),
            })
            # Log aggregate best metrics on the parent
            mlflow.log_metric("best_score", baseline_result.get("score") or 0.0)

            for row in candidate_results:
                model_name = row.get("model", "unknown")
                with mlflow.start_run(run_name=f"baseline_{model_name}", nested=True, tags={
                    "run_type":  "baseline_candidate",
                    "model":     model_name,
                    "strategy":  row.get("imbalance_strategy", "default"),
                }):
                    metrics = {}
                    for key in ("accuracy", "f1_weighted", "r2", "mae", "rmse"):
                        val = row.get(key)
                        if val is not None:
                            metrics[key] = float(val)
                    if metrics:
                        mlflow.log_metrics(metrics)

            return parent_run.info.run_id
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HPO logging
# ─────────────────────────────────────────────────────────────────────────────

def log_hpo_start(
    experiment_name: str,
    tracking_uri: str,
    state: Dict,
    tuning_mode: str,
    cv_folds: int,
    n_trials: int,
    model_ids: list,
) -> Optional[str]:
    """
    Opens the HPO parent MLflow run (without a context manager so it stays
    open across the trial loop). Returns run_id. Caller must call end_hpo_run().
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        setup_mlflow_experiment(tracking_uri, experiment_name)
        run = mlflow.start_run(
            run_name=f"hpo_{tuning_mode}",
            tags={
                "run_type":    "hpo_parent",
                "tuning_mode": tuning_mode,
                "target_column": state.get("target_column", ""),
                "problem_type":  state.get("problem_type", ""),
            },
        )
        mlflow.log_params({
            "dataset_path":            state.get("dataset_path", ""),
            "target_column":           state.get("target_column", ""),
            "problem_type":            state.get("problem_type", ""),
            "tuning_mode":             tuning_mode,
            "cv_folds":                cv_folds,
            "optuna_trials_per_model": n_trials,
            "candidate_models":        ",".join(model_ids),
        })
        return run.info.run_id
    except Exception:
        return None


def log_tuning_trial(
    parent_run_id: str,
    model_id: str,
    trial_index: int,
    params: Dict,
    scores: Dict,
    strategy: str,
    metric_name: str,
) -> None:
    """Log a single Optuna trial as a nested child run under parent_run_id."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(
            run_name=f"trial_{model_id}_{trial_index}",
            nested=True,
            tags={
                "run_type":    "hpo_trial",
                "model_id":    model_id,
                "strategy":    strategy,
                "trial_index": str(trial_index),
            },
        ):
            # Log hyperparameters — convert values to strings to avoid type issues
            safe_params = {k: str(v) for k, v in params.items()}
            if safe_params:
                mlflow.log_params(safe_params)

            metrics = {}
            for key in ("cv_score", "primary_score", "accuracy", "f1_weighted", "r2", "mae", "rmse"):
                val = scores.get(key)
                if val is not None:
                    metrics[key] = float(val)
            # Rename primary_score to holdout_primary_score for clarity
            if "primary_score" in metrics:
                metrics["holdout_primary_score"] = metrics.pop("primary_score")
            if metrics:
                mlflow.log_metrics(metrics)
    except Exception:
        pass


def end_hpo_run(run_id: str, best_score: Optional[float], best_model_id: Optional[str]) -> None:
    """Log final summary metrics on the parent HPO run and end it."""
    if not MLFLOW_AVAILABLE or not run_id:
        return
    try:
        with mlflow.start_run(run_id=run_id):
            if best_score is not None:
                mlflow.log_metric("best_score", float(best_score))
            if best_model_id:
                mlflow.set_tag("best_model_id", best_model_id)
        mlflow.end_run()
    except Exception:
        try:
            mlflow.end_run()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Reuse: fetch best params from a prior completed HPO run
# ─────────────────────────────────────────────────────────────────────────────

def fetch_best_hpo_params(experiment_name: str, tracking_uri: str) -> Optional[Dict[str, Any]]:
    """
    Searches MLflow for the best completed HPO parent run for this experiment.
    Returns {model_id: {params, cv_score, score, strategy, run_id}} or None.

    Logic:
    1. Find finished runs tagged run_type=hpo_parent, ordered by best_score DESC.
    2. Take the top run; fetch all its hpo_trial child runs.
    3. Group by model_id, pick child with highest cv_score per model.
    4. Return that mapping so the agent can reconstruct estimators without Optuna.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        setup_mlflow_experiment(tracking_uri, experiment_name)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None

        # Find completed HPO parent runs
        parent_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.run_type = 'hpo_parent' AND attributes.status = 'FINISHED'",
            order_by=["metrics.best_score DESC", "attributes.start_time DESC"],
            max_results=5,
        )
        if not parent_runs:
            return None

        best_parent = parent_runs[0]
        parent_run_id = best_parent.info.run_id

        # Fetch all trial child runs under that parent
        child_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=(
                f"tags.`mlflow.parentRunId` = '{parent_run_id}' "
                "AND tags.run_type = 'hpo_trial'"
            ),
            order_by=["metrics.cv_score DESC"],
            max_results=1000,
        )
        if not child_runs:
            return None

        # Group by model_id, keep best cv_score per model
        best_per_model: Dict[str, Any] = {}
        for run in child_runs:
            model_id = run.data.tags.get("model_id", "")
            if not model_id:
                continue
            cv_score = run.data.metrics.get("cv_score", 0.0)
            if model_id not in best_per_model or cv_score > best_per_model[model_id]["cv_score"]:
                # Reconstruct params — stored as strings, try numeric conversion
                raw_params = dict(run.data.params)
                converted = {}
                for k, v in raw_params.items():
                    try:
                        converted[k] = int(v)
                    except (ValueError, TypeError):
                        try:
                            converted[k] = float(v)
                        except (ValueError, TypeError):
                            # Handle tuple strings like "(64, 32)"
                            if v.startswith("(") and v.endswith(")"):
                                try:
                                    converted[k] = eval(v)  # noqa: S307 — safe: MLflow-stored tuples only
                                except Exception:
                                    converted[k] = v
                            else:
                                converted[k] = v

                best_per_model[model_id] = {
                    "params":   converted,
                    "cv_score": cv_score,
                    "score":    run.data.metrics.get("holdout_primary_score", 0.0),
                    "strategy": run.data.tags.get("strategy", "default"),
                    "run_id":   run.info.run_id,
                }

        return best_per_model if best_per_model else None
    except Exception:
        return None
