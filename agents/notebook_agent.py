# agents/notebook_agent.py
#
# Generates a self-contained, Kaggle-ready Jupyter notebook from agent insights.
# The notebook covers: EDA → Preprocessing → Baseline → Tuning → Eval → Submission.

from __future__ import annotations

import json
import os
import textwrap

import nbformat as nbf


# ─────────────────────────────────────────────────────────────────────────────
# Cell builders
# ─────────────────────────────────────────────────────────────────────────────

def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_preprocessing(preprocessing_report: dict) -> dict:
    dropped     = list(preprocessing_report.get("dropped_columns", []))
    transformed = preprocessing_report.get("transformed_columns", [])
    encoded     = preprocessing_report.get("encoded_columns", {"frequency": [], "one_hot": [], "target_encode": []})
    fe_steps    = preprocessing_report.get("fe_steps_applied", [])

    log_cols        = [c.split(":")[0] for c in transformed if ":log1p" in c]
    yj_cols         = [c.split(":")[0] for c in transformed if ":yeo_johnson" in c]
    winsorized_cols = [c.split(":")[0] for c in transformed if ":winsorized" in c]
    binned_cols     = [c.split(":")[0] for c in transformed if ":bin_quantile" in c]
    freq_cols       = list(encoded.get("frequency", []))
    ohe_cols        = list(encoded.get("one_hot", []))
    te_cols         = list(encoded.get("target_encode", []))

    # Columns that were kept alive to create interaction features, then dropped after.
    # These must be dropped POST-interaction in the notebook, not upfront.
    interaction_src  = set(preprocessing_report.get("interaction_source_columns", []))
    interaction_out  = set(preprocessing_report.get("interaction_output_columns", []))
    # Interaction output columns must never appear in any drop list
    dropped          = [c for c in dropped if c not in interaction_out]
    dropped_late     = [c for c in dropped if c in interaction_src]
    dropped_early    = [c for c in dropped if c not in interaction_src]

    # Remove from all transform/encode lists any column already scheduled for dropping
    dropped_set     = set(dropped)
    log_cols        = [c for c in log_cols        if c not in dropped_set]
    yj_cols         = [c for c in yj_cols         if c not in dropped_set]
    winsorized_cols = [c for c in winsorized_cols if c not in dropped_set]
    binned_cols     = [c for c in binned_cols     if c not in dropped_set]
    freq_cols       = [c for c in freq_cols       if c not in dropped_set]
    ohe_cols        = [c for c in ohe_cols        if c not in dropped_set]
    te_cols         = [c for c in te_cols         if c not in dropped_set]

    # Interaction features created
    interactions = [
        s for s in fe_steps if s.get("action") == "interaction"
    ]
    # Cyclical datetime columns
    cyclical_cols = [
        s["col"] for s in fe_steps if s.get("method") == "cyclical"
    ]

    return {
        "dropped":          dropped_early,
        "dropped_late":     dropped_late,
        "log_cols":         log_cols,
        "yj_cols":          yj_cols,
        "winsorized_cols":  winsorized_cols,
        "binned_cols":      binned_cols,
        "freq_cols":        freq_cols,
        "ohe_cols":         ohe_cols,
        "te_cols":          te_cols,
        "interactions":     interactions,
        "cyclical_cols":    cyclical_cols,
        "fe_steps":         fe_steps,
        "llm_fe_used":      preprocessing_report.get("llm_fe_used", False),
    }


def _top_features(stats_report: dict, top_n: int = 8) -> list:
    ranked = stats_report.get("ranked_features", [])
    if ranked:
        return [r["feature"] for r in ranked[:top_n]]
    return stats_report.get("significant_after_fdr", [])[:top_n]


def _get_model_info(model_id: str, problem_type: str, best_params: dict) -> dict:
    """Return class name, import line, whether needs scaler, and param string."""
    clean = {k.replace("model__", ""): v for k, v in (best_params or {}).items()}

    CLS = {
        "classification": {
            "logistic_regression":               ("LogisticRegression",              "from sklearn.linear_model import LogisticRegression",             True),
            "ridge_classifier":                  ("RidgeClassifier",                 "from sklearn.linear_model import RidgeClassifier",                True),
            "linear_svc":                        ("LinearSVC",                       "from sklearn.svm import LinearSVC",                               True),
            "svc_rbf":                           ("SVC",                             "from sklearn.svm import SVC",                                     True),
            "knn_classifier":                    ("KNeighborsClassifier",            "from sklearn.neighbors import KNeighborsClassifier",              True),
            "gaussian_nb":                       ("GaussianNB",                      "from sklearn.naive_bayes import GaussianNB",                      False),
            "decision_tree_classifier":          ("DecisionTreeClassifier",          "from sklearn.tree import DecisionTreeClassifier",                 False),
            "random_forest_classifier":          ("RandomForestClassifier",          "from sklearn.ensemble import RandomForestClassifier",             False),
            "extra_trees_classifier":            ("ExtraTreesClassifier",            "from sklearn.ensemble import ExtraTreesClassifier",               False),
            "gradient_boosting_classifier":      ("GradientBoostingClassifier",      "from sklearn.ensemble import GradientBoostingClassifier",         False),
            "hist_gradient_boosting_classifier": ("HistGradientBoostingClassifier",  "from sklearn.ensemble import HistGradientBoostingClassifier",     False),
            "ada_boost_classifier":              ("AdaBoostClassifier",              "from sklearn.ensemble import AdaBoostClassifier",                 False),
            "bagging_classifier":                ("BaggingClassifier",               "from sklearn.ensemble import BaggingClassifier",                  False),
            "mlp_classifier":                    ("MLPClassifier",                   "from sklearn.neural_network import MLPClassifier",                True),
            "xgboost_classifier":                ("XGBClassifier",                   "from xgboost import XGBClassifier",                               False),
            "lightgbm_classifier":               ("LGBMClassifier",                  "from lightgbm import LGBMClassifier",                             False),
        },
        "regression": {
            "linear_regression":                 ("LinearRegression",                "from sklearn.linear_model import LinearRegression",               True),
            "ridge_regressor":                   ("Ridge",                           "from sklearn.linear_model import Ridge",                          True),
            "lasso_regressor":                   ("Lasso",                           "from sklearn.linear_model import Lasso",                          True),
            "elasticnet_regressor":              ("ElasticNet",                      "from sklearn.linear_model import ElasticNet",                     True),
            "linear_svr":                        ("LinearSVR",                       "from sklearn.svm import LinearSVR",                               True),
            "svr_rbf":                           ("SVR",                             "from sklearn.svm import SVR",                                     True),
            "knn_regressor":                     ("KNeighborsRegressor",             "from sklearn.neighbors import KNeighborsRegressor",               True),
            "decision_tree_regressor":           ("DecisionTreeRegressor",           "from sklearn.tree import DecisionTreeRegressor",                  False),
            "random_forest_regressor":           ("RandomForestRegressor",           "from sklearn.ensemble import RandomForestRegressor",              False),
            "extra_trees_regressor":             ("ExtraTreesRegressor",             "from sklearn.ensemble import ExtraTreesRegressor",                False),
            "gradient_boosting_regressor":       ("GradientBoostingRegressor",       "from sklearn.ensemble import GradientBoostingRegressor",          False),
            "hist_gradient_boosting_regressor":  ("HistGradientBoostingRegressor",   "from sklearn.ensemble import HistGradientBoostingRegressor",      False),
            "ada_boost_regressor":               ("AdaBoostRegressor",               "from sklearn.ensemble import AdaBoostRegressor",                  False),
            "bagging_regressor":                 ("BaggingRegressor",                "from sklearn.ensemble import BaggingRegressor",                   False),
            "mlp_regressor":                     ("MLPRegressor",                    "from sklearn.neural_network import MLPRegressor",                 True),
            "xgboost_regressor":                 ("XGBRegressor",                    "from xgboost import XGBRegressor",                                False),
            "lightgbm_regressor":                ("LGBMRegressor",                   "from lightgbm import LGBMRegressor",                              False),
        },
    }

    defaults = {
        "classification": ("HistGradientBoostingClassifier", "from sklearn.ensemble import HistGradientBoostingClassifier", False),
        "regression":     ("HistGradientBoostingRegressor",  "from sklearn.ensemble import HistGradientBoostingRegressor",  False),
    }

    class_name, import_line, needs_scaling = CLS.get(problem_type, CLS["classification"]).get(
        model_id, defaults.get(problem_type, defaults["classification"])
    )

    # Fixed params we always set explicitly
    _SKIP = {"random_state", "n_jobs", "max_iter", "early_stopping", "eval_metric"}
    tuned = {k: v for k, v in clean.items() if k not in _SKIP}

    # Build instantiation string
    parts = []
    for k, v in tuned.items():
        parts.append(f"{k}={repr(v)}")
    parts.append("random_state=RANDOM_STATE")
    if class_name in {
        "RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier",
        "RandomForestRegressor",  "ExtraTreesRegressor",  "BaggingRegressor",
    }:
        parts.append("n_jobs=-1")

    instantiation = f"{class_name}({', '.join(parts)})"

    return {
        "class_name": class_name,
        "import_line": import_line,
        "needs_scaling": needs_scaling,
        "instantiation": instantiation,
        "clean_params": clean,
    }


def _optuna_suggest_block(model_id: str, problem_type: str) -> str:
    """Return the trial.suggest_* block + model constructor for Optuna."""
    is_cls = problem_type == "classification"

    if "random_forest" in model_id or "extra_trees" in model_id:
        cls = "RandomForestClassifier" if is_cls else "RandomForestRegressor"
        mod = "from sklearn.ensemble import " + cls
        return f"""\
    params = {{
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":       trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }}
    model = {cls}(**params, random_state=RANDOM_STATE, n_jobs=-1)"""

    if "hist_gradient" in model_id:
        cls = "HistGradientBoostingClassifier" if is_cls else "HistGradientBoostingRegressor"
        return f"""\
    params = {{
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":      trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, step=8),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 5, 50, step=5),
    }}
    model = {cls}(**params, random_state=RANDOM_STATE)"""

    if "gradient_boosting" in model_id:
        cls = "GradientBoostingClassifier" if is_cls else "GradientBoostingRegressor"
        return f"""\
    params = {{
        "n_estimators":  trial.suggest_int("n_estimators", 50, 300, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":     trial.suggest_int("max_depth", 2, 6),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
    }}
    model = {cls}(**params, random_state=RANDOM_STATE)"""

    if "logistic_regression" in model_id or "ridge_classifier" in model_id:
        cls = "LogisticRegression" if is_cls else "Ridge"
        wrap = "Pipeline([('scaler', StandardScaler()), ('model', {cls}(C=C, max_iter=2000, random_state=RANDOM_STATE))])" if is_cls else f"Ridge(alpha=1.0/C, random_state=RANDOM_STATE)"
        return f"""\
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    model = {wrap}"""

    # fallback
    cls = "HistGradientBoostingClassifier" if is_cls else "HistGradientBoostingRegressor"
    return f"""\
    params = {{
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, step=8),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 5, 50, step=5),
    }}
    model = {cls}(**params, random_state=RANDOM_STATE)"""


# ─────────────────────────────────────────────────────────────────────────────
# Section generators (each returns a list of cells)
# ─────────────────────────────────────────────────────────────────────────────

def _section_mlflow_info(state: dict) -> list:
    """Insert an MLflow experiment reference cell if the pipeline logged to MLflow."""
    hpo_run  = state.get("mlflow_hpo_run_id", "")
    base_run = state.get("mlflow_baseline_run_id", "")
    exp_name = state.get("mlflow_experiment_name", "")
    mode     = state.get("tuning_mode", "")
    uri      = state.get("mlflow_tracking_uri", "./mlruns")

    if not (hpo_run or base_run):
        return []

    lines = ["# MLflow Experiment Tracking", ""]
    if exp_name:
        lines.append(f"- **Experiment:** `{exp_name}`")
    if base_run:
        lines.append(f"- **Baseline Run ID:** `{base_run}`")
    if hpo_run:
        lines.append(f"- **HPO Run ID:** `{hpo_run}`")
    if mode:
        lines.append(f"- **Tuning Mode:** `{mode}`")
    lines += [
        "",
        "To view results in the MLflow UI:",
        f"```bash",
        f"mlflow ui --backend-store-uri {uri}",
        f"```",
        "Then open http://localhost:5000 in your browser.",
    ]
    return [_md("\n".join(lines))]


def _section_title(state: dict) -> list:
    target = state.get("target_column", "target")
    problem_type = state.get("problem_type", "classification").title()
    objective = state.get("objective", "")
    dataset_path = state.get("dataset_path", "data.csv")

    profiling = state.get("profiling_report", {})
    shape = profiling.get("dataset_shape", {})
    n_rows = shape.get("rows", "N/A")
    n_cols = shape.get("columns", "N/A")
    quality = profiling.get("data_quality_score", {})
    q_score = quality.get("overall_score", "N/A")
    q_grade = quality.get("grade", "N/A")

    try:
        n_rows_fmt = f"{n_rows:,}"
    except Exception:
        n_rows_fmt = str(n_rows)

    return [_md(f"""\
# Expert ML Pipeline: {problem_type} — `{target}`

> **Objective:** {objective}

| | |
|---|---|
| **Dataset** | `{dataset_path}` |
| **Target** | `{target}` |
| **Problem type** | {problem_type} |
| **Rows / Columns** | {n_rows_fmt} / {n_cols} |
| **Data quality** | {q_score}/100 (Grade {q_grade}) |

---
*Auto-generated by the Agentic ML Pipeline. All preprocessing decisions, model selection,
and hyperparameter tuning are driven by data-analysis agents. Run cells top-to-bottom.*
""")]


def _section_setup(state: dict) -> list:
    problem_type = state.get("problem_type", "classification")
    is_cls = problem_type == "classification"

    if is_cls:
        model_imports = """\
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier"""
        xgb_import  = "from xgboost import XGBClassifier"
        lgbm_import = "from lightgbm import LGBMClassifier"
    else:
        model_imports = """\
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor,
)
from sklearn.neural_network import MLPRegressor"""
        xgb_import  = "from xgboost import XGBRegressor"
        lgbm_import = "from lightgbm import LGBMRegressor"

    return [
        _md("# Setup\n\nImport all libraries needed for the full ML pipeline."),
        _code(f"""\
import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.pipeline import Pipeline
{model_imports}

try:
    {xgb_import}
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    {lgbm_import}
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("optuna not installed — run: pip install optuna")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{{x:.4f}}")

RANDOM_STATE = 42
print("Environment ready.")
"""),
    ]


def _section_data_loading(state: dict) -> list:
    target = state.get("target_column", "target")
    dataset_path = state.get("dataset_path", "data.csv")
    problem_type = state.get("problem_type", "classification")

    base = os.path.basename(dataset_path)
    if "train" in base.lower():
        test_path = dataset_path.replace(base, base.lower().replace("train", "test"))
    else:
        root, ext = os.path.splitext(dataset_path)
        test_path = root + "_test" + ext

    return [
        _md("# 1. Data Loading & Initial Exploration"),
        _code(f"""\
TARGET       = {repr(target)}
TRAIN_PATH   = r{repr(dataset_path)}
TEST_PATH    = r{repr(test_path)}   # update if your test file has a different path
PROBLEM_TYPE = {repr(problem_type)}

df = pd.read_csv(TRAIN_PATH)
print(f"Training set shape: {{df.shape}}")
print(f"Target column  : {{TARGET}}")
print(f"Problem type   : {{PROBLEM_TYPE}}")
df.head()
"""),
        _code("""\
# Overview: dtypes, missing values, basic stats
print("=== Column dtypes ===")
print(df.dtypes.value_counts())

print("\\n=== Missing values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct (%)": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("count", ascending=False)

if not missing_df.empty:
    display(missing_df)
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.35)))
    missing_df["pct (%)"].plot(kind="barh", ax=ax, color="#ef4444")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values per Column", fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("  No missing values found!")

print("\\n=== Basic statistics ===")
display(df.describe(include="all").T)
"""),
    ]


def _section_eda(state: dict) -> list:
    target = state.get("target_column", "target")
    problem_type = state.get("problem_type", "classification")
    eda_report = state.get("eda_report", {})

    insights = eda_report.get("insights", [])[:8]
    insights_md = "\n".join(f"- {i}" for i in insights) if insights else "- No insights available."

    if problem_type == "classification":
        target_plot = """\
tc = df[TARGET].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(tc.index.astype(str), tc.values, color="#6366f1")
axes[0].set_title(f"Target distribution: {TARGET}", fontweight="bold")
axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
for i, v in enumerate(tc.values):
    axes[0].text(i, v + 0.3, f"{v:,}\\n({v/len(df)*100:.1f}%)", ha="center", fontsize=8)

axes[1].pie(tc.values, labels=tc.index.astype(str),
            autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("husl", len(tc)))
axes[1].set_title("Class balance", fontweight="bold")

plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()
print(df[TARGET].value_counts())"""
    else:
        target_plot = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df[TARGET].dropna(), bins=50, color="#6366f1", edgecolor="white", alpha=0.8)
axes[0].set_title(f"Distribution: {TARGET}", fontweight="bold")
axes[0].set_xlabel(TARGET); axes[0].set_ylabel("Count")

axes[1].boxplot(df[TARGET].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor="#6366f1", alpha=0.7))
axes[1].set_title(f"Spread: {TARGET}", fontweight="bold")
axes[1].set_ylabel(TARGET)

plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()
print(df[TARGET].describe())"""

    return [
        _md("# 2. Exploratory Data Analysis\n\nVisualise the target variable and every feature in the dataset."),
        _code(f"""\
# ── Target distribution ──────────────────────────────────────────
{target_plot}
"""),
        _code("""\
# ── All numeric feature distributions ────────────────────────────
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
N_PER_ROW = 4
n_rows = math.ceil(len(num_cols) / N_PER_ROW) if num_cols else 1

if num_cols:
    fig, axes = plt.subplots(n_rows, N_PER_ROW, figsize=(16, n_rows * 4))
    axes_flat = np.array(axes).flatten() if (n_rows > 1 or N_PER_ROW > 1) else [axes]

    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, density=True, alpha=0.65, color="#6366f1", edgecolor="white")
        try:
            from scipy.stats import gaussian_kde
            if len(data) > 1 and data.std() > 0:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 200)
                ax.plot(xs, kde(xs), color="#ef4444", lw=2)
        except Exception:
            pass
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_xlabel(col); ax.set_ylabel("Density")

    for j in range(len(num_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("No numeric features found.")
"""),
        _code("""\
# ── All categorical feature distributions ────────────────────────
cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != TARGET]
N_PER_ROW = 4
n_rows = math.ceil(len(cat_cols) / N_PER_ROW) if cat_cols else 1

if cat_cols:
    fig, axes = plt.subplots(n_rows, N_PER_ROW, figsize=(16, n_rows * 4))
    axes_flat = np.array(axes).flatten() if (n_rows > 1 or N_PER_ROW > 1) else [axes]

    for i, col in enumerate(cat_cols):
        ax = axes_flat[i]
        vc = df[col].value_counts().head(10)
        ax.bar(range(len(vc)), vc.values, color="#6366f1", alpha=0.8)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel("Count")

    for j in range(len(cat_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("No categorical features found.")
"""),
        _code("""\
# ── Correlation heatmap ───────────────────────────────────────────
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols_all) >= 2:
    target_in_num = TARGET in num_cols_all
    if target_in_num:
        tc_abs = df[num_cols_all].corr()[TARGET].abs().sort_values(ascending=False)
        plot_cols = tc_abs.head(20).index.tolist()
    else:
        plot_cols = num_cols_all[:20]

    corr = df[plot_cols].corr()
    fig, axes = plt.subplots(1, 2 if target_in_num else 1,
                              figsize=(18 if target_in_num else 10, 7))
    if not target_in_num:
        axes = [axes]

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                annot=len(plot_cols) <= 12, fmt=".2f",
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")

    if target_in_num:
        tc = df[num_cols_all].corr()[TARGET].drop(TARGET).abs() \\
               .sort_values(ascending=False).head(15)
        colors = ["#22c55e" if v > 0.3 else "#f59e0b" if v > 0.1 else "#64748b"
                  for v in tc.values]
        axes[1].barh(tc.index, tc.values, color=colors)
        axes[1].set_title(f"Correlation with {{TARGET}}", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Absolute Correlation")
        axes[1].axvline(0.3, color="green", ls="--", alpha=0.7, label="Strong (0.3)")
        axes[1].axvline(0.1, color="orange", ls="--", alpha=0.7, label="Moderate (0.1)")
        axes[1].legend()

    plt.tight_layout(); plt.show()
"""),
        _md(f"""\
## Key Insights from Agent Analysis

{insights_md}
"""),
    ]


def _section_statistical_analysis(state: dict) -> list:
    stats_report  = state.get("stats_report", {})
    problem_type  = state.get("problem_type", "classification")

    significant   = stats_report.get("significant_after_fdr", [])
    insignificant = stats_report.get("insignificant", [])
    large_effect  = stats_report.get("large_effect_features", [])
    medium_effect = stats_report.get("medium_effect_features", [])

    sig_bullets = "\n".join(f"  - `{f}`" for f in significant[:15]) or "  - None identified"
    large_str   = ", ".join(f"`{f}`" for f in large_effect[:8]) or "None"
    medium_str  = ", ".join(f"`{f}`" for f in medium_effect[:8]) or "None"

    is_cls = problem_type == "classification"
    num_test_comment = "Mann-Whitney U (2-class) or Kruskal-Wallis (multi-class)" if is_cls else "Spearman correlation"
    cat_test_comment = "Chi-square contingency test" if is_cls else "Kruskal-Wallis H-test"

    return [
        _md("# 3. Statistical Feature Significance\n\n"
            "Run statistical tests on every feature to measure its relationship with the target. "
            "Benjamini-Hochberg FDR correction is applied to control false discoveries."),
        _code(f"""\
# ── Run statistical tests on all features ────────────────────────
# Numeric vs target : {num_test_comment}
# Categorical vs target : {cat_test_comment}
# FDR correction : Benjamini-Hochberg

feature_cols = [c for c in df.columns if c != TARGET]
num_feats = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_feats = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
n_classes = df[TARGET].nunique()

def _bh_correction(p_values):
    # Benjamini-Hochberg FDR correction (no external dependency)
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    idx   = np.argsort(p)
    adj   = p[idx] * n / (np.arange(1, n + 1))
    # Enforce monotonicity right-to-left
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result = np.empty_like(adj)
    result[idx] = adj
    return result

stat_rows = []

for col in num_feats:
    sub = df[[col, TARGET]].dropna()
    x, y = sub[col].values, sub[TARGET].values
    try:
        if PROBLEM_TYPE == "classification":
            classes = np.unique(y)
            groups  = [x[y == c] for c in classes if (y == c).sum() >= 3]
            if len(groups) < 2:
                continue
            if len(groups) == 2:
                stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                test_name = "Mann-Whitney U"
            else:
                stat, p = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"
        else:
            stat, p = stats.spearmanr(x, y)
            test_name = "Spearman r"
        stat_rows.append({{"feature": col, "dtype": "numeric", "test": test_name,
                           "statistic": round(float(stat), 4), "p_value": float(p)}})
    except Exception as e:
        print(f"  Skipped {{col}}: {{e}}")

for col in cat_feats:
    sub = df[[col, TARGET]].dropna()
    try:
        if PROBLEM_TYPE == "classification":
            ct = pd.crosstab(sub[col], sub[TARGET])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            stat, p, _, _ = stats.chi2_contingency(ct)
            test_name = "Chi-square"
        else:
            groups = [sub[TARGET][sub[col] == v].values
                      for v in sub[col].unique() if (sub[col] == v).sum() >= 3]
            if len(groups) < 2:
                continue
            stat, p = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        stat_rows.append({{"feature": col, "dtype": "categorical", "test": test_name,
                           "statistic": round(float(stat), 4), "p_value": float(p)}})
    except Exception as e:
        print(f"  Skipped {{col}}: {{e}}")

stat_df = pd.DataFrame(stat_rows).sort_values("p_value").reset_index(drop=True)

if not stat_df.empty:
    stat_df["p_value_fdr"] = _bh_correction(stat_df["p_value"].values)
    stat_df["significant"] = stat_df["p_value_fdr"] < 0.05
    print(f"Features tested          : {{len(stat_df)}}")
    print(f"Significant (p_fdr<0.05) : {{stat_df['significant'].sum()}}")
    display(stat_df.style.format({{"p_value": "{{:.4e}}", "p_value_fdr": "{{:.4e}}",
                                   "statistic": "{{:.4f}}"}}))
else:
    print("No features could be tested.")
    stat_df = pd.DataFrame(columns=["feature", "dtype", "test", "statistic", "p_value", "p_value_fdr", "significant"])
"""),
        _code("""\
# ── Compute effect sizes, MI scores and rank features ────────────
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from matplotlib.patches import Patch

def _cramers_v_nb(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, c = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0

def _cohens_d_nb(g1, g2):
    pooled = np.sqrt((np.std(g1) ** 2 + np.std(g2) ** 2) / 2)
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 0 else 0.0

def _epsilon_sq_nb(H, n, k):
    return float((H - k + 1) / (n - k)) if n > k else 0.0

def _effect_label_nb(size, metric):
    a = abs(size)
    if metric in ("cramers_v", "epsilon_squared", "eta_squared"):
        if a >= 0.50: return "large"
        if a >= 0.30: return "medium"
        if a >= 0.10: return "small"
        return "negligible"
    if metric == "cohens_d":
        if a >= 0.80: return "large"
        if a >= 0.50: return "medium"
        if a >= 0.20: return "small"
        return "negligible"
    # spearman / pearson r
    if a >= 0.50: return "large"
    if a >= 0.30: return "medium"
    if a >= 0.10: return "small"
    return "negligible"

effect_map = {}
for _, row in stat_df.iterrows():
    col  = row["feature"]
    sub  = df[[col, TARGET]].dropna()
    x, y = sub[col].values, sub[TARGET].values
    test = row["test"]
    try:
        if test == "Mann-Whitney U":
            classes = np.unique(y)
            g1, g2  = x[y == classes[0]], x[y == classes[1]]
            es = _cohens_d_nb(g1, g2)
            effect_map[col] = (round(es, 4), "cohens_d")
        elif test == "Kruskal-Wallis":
            if PROBLEM_TYPE == "classification":
                H, n, k = row["statistic"], len(x), len(np.unique(y))
            else:
                H, n, k = row["statistic"], len(y), len(np.unique(x))
            effect_map[col] = (round(_epsilon_sq_nb(H, n, k), 4), "epsilon_squared")
        elif test == "Chi-square":
            ct = pd.crosstab(sub[col], sub[TARGET])
            effect_map[col] = (round(_cramers_v_nb(ct), 4), "cramers_v")
        elif test in ("Spearman r", "Pearson r"):
            effect_map[col] = (round(abs(float(row["statistic"])), 4), "spearman_r")
        else:
            effect_map[col] = (0.0, "unknown")
    except Exception:
        effect_map[col] = (0.0, "unknown")

# MI scores (normalised to [0, 1] for ranking)
X_mi = df[feature_cols].copy()
for c in cat_feats:
    X_mi[c] = X_mi[c].astype("category").cat.codes
X_mi = X_mi.fillna(X_mi.median(numeric_only=True))
if PROBLEM_TYPE == "classification":
    mi_raw = mutual_info_classif(X_mi, df[TARGET], random_state=42)
else:
    mi_raw = mutual_info_regression(X_mi, df[TARGET], random_state=42)
mi_map = dict(zip(feature_cols, mi_raw))
max_mi = max(mi_map.values()) if mi_map else 1.0
if max_mi == 0:
    max_mi = 1.0

ranked_features = []
for _, row in stat_df.iterrows():
    col        = row["feature"]
    es, metric = effect_map.get(col, (0.0, "unknown"))
    mi_val     = float(mi_map.get(col, 0.0))
    mi_norm    = mi_val / max_mi
    combined   = round(max(abs(es), mi_norm), 4)
    ranked_features.append({
        "feature":               col,
        "test":                  row["test"],
        "p_value":               round(float(row["p_value"]), 6),
        "p_adjusted":            round(float(row["p_value_fdr"]), 6),
        "effect_size":           es,
        "effect_label":          _effect_label_nb(es, metric),
        "mi_score":              round(mi_val, 4),
        "combined_score":        combined,
        "significant":           bool(row["significant"]),
        "significant_after_fdr": bool(row["significant"]),
    })

ranked_features.sort(key=lambda r: (-int(r["significant_after_fdr"]), -r["combined_score"]))
large_effect  = [r["feature"] for r in ranked_features if r["effect_label"] == "large"]
medium_effect = [r["feature"] for r in ranked_features if r["effect_label"] == "medium"]
print(f"Large effect features  : {large_effect  or 'none'}")
print(f"Medium effect features : {medium_effect or 'none'}")
"""),
        _code("""\
# ── Visualise effect size and significance ────────────────────────
if ranked_features:
    rdf = pd.DataFrame(ranked_features)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(rdf) * 0.4)))

    if "effect_size" in rdf.columns:
        rdf_s = rdf.sort_values("effect_size", ascending=True).tail(15)
        colors = [
            "#22c55e" if r in large_effect else
            "#f59e0b" if r in medium_effect else "#64748b"
            for r in rdf_s["feature"].tolist()
        ]
        axes[0].barh(rdf_s["feature"].astype(str), rdf_s["effect_size"], color=colors)
        axes[0].set_title("Feature Effect Size", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Effect Size")
        axes[0].legend(handles=[
            Patch(color="#22c55e", label="Large"),
            Patch(color="#f59e0b", label="Medium"),
            Patch(color="#64748b", label="Small/Negligible"),
        ])

    if "p_adjusted" in rdf.columns:
        rdf["neg_log_p"] = -np.log10(rdf["p_adjusted"].clip(lower=1e-300))
        rdf_s2 = rdf.sort_values("neg_log_p", ascending=True).tail(15)
        c2 = ["#22c55e" if p < 0.001 else "#f59e0b" if p < 0.01 else "#6366f1"
               for p in rdf_s2["p_adjusted"]]
        axes[1].barh(rdf_s2["feature"].astype(str), rdf_s2["neg_log_p"], color=c2)
        axes[1].axvline(-np.log10(0.05), color="red", ls="--", label="FDR = 0.05")
        axes[1].set_title("-log₁₀(FDR p-value)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("-log₁₀(p_fdr)")
        axes[1].legend()

    plt.suptitle("Statistical Feature Significance", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
"""),
        _md(f"""\
## Interpretation

Based on the statistical tests above:

**Significant features ({len(significant)} features, p_fdr < 0.05):**

{sig_bullets}

**Large effect size:** {large_str}
**Medium effect size:** {medium_str}
**Insignificant / deprioritised:** {len(insignificant)} features

Features with large effect sizes and low FDR-corrected p-values are the most informative
predictors. Use these findings to guide feature selection and engineering in the next section.
"""),
    ]


def _fe_steps_md(fe_steps: list) -> str:
    """Render a human-readable summary of FE steps for the notebook markdown cell."""
    if not fe_steps:
        return "  - Standard imputation + encoding pipeline applied."
    lines = []
    for s in fe_steps:
        col    = s.get("col", "?")
        action = s.get("action", "")
        method = s.get("method") or ""
        reason = s.get("reason", "")
        method_str = f" ({method})" if method else ""
        lines.append(f"  - **{col}** → `{action}{method_str}`: {reason}")
    return "\n".join(lines)


def _section_preprocessing(state: dict) -> list:
    preprocessing_report = state.get("preprocessing_report", {})
    target = state.get("target_column", "target")

    cfg = _parse_preprocessing(preprocessing_report)
    dropped       = cfg["dropped"]
    dropped_late  = cfg["dropped_late"]
    log_cols      = cfg["log_cols"]
    yj_cols       = cfg["yj_cols"]
    winsorized    = cfg["winsorized_cols"]
    binned_cols   = cfg["binned_cols"]
    freq_cols     = cfg["freq_cols"]
    ohe_cols      = cfg["ohe_cols"]
    te_cols       = cfg["te_cols"]
    interactions  = cfg["interactions"]
    cyclical_cols = cfg["cyclical_cols"]
    fe_steps      = cfg["fe_steps"]
    llm_fe_used   = cfg["llm_fe_used"]

    steps   = preprocessing_report.get("steps", [])
    n_orig  = preprocessing_report.get("original_feature_count", "?")
    n_final = preprocessing_report.get("selected_feature_count", "?")
    n_int   = preprocessing_report.get("interactions_created", 0)

    steps_md  = "\n".join(f"  - {s.replace('_', ' ').title()}" for s in steps) or "  - Standard pipeline"
    fe_md     = _fe_steps_md(fe_steps)
    llm_note  = "*(LLM-directed feature engineering)*" if llm_fe_used else "*(rule-based feature engineering)*"

    winsorize_cols_all = list(set(winsorized + log_cols))

    interaction_code = ""
    if interactions:
        lines = []
        for ix in interactions:
            ca, cb, itype = ix.get("col_a"), ix.get("col_b"), ix.get("method", "ratio")
            if itype == "ratio":
                lines.append(f'    X["{ca}_div_{cb}"] = X["{ca}"] / X["{cb}"].replace(0, np.nan).fillna(0)')
            elif itype == "product":
                lines.append(f'    X["{ca}_x_{cb}"] = X["{ca}"] * X["{cb}"]')
            elif itype == "difference":
                lines.append(f'    X["{ca}_minus_{cb}"] = X["{ca}"] - X["{cb}"]')
        interaction_code = "\n    # ── Interaction features (LLM-recommended) ──\n" + "\n".join(lines)

    te_code = ""
    if te_cols:
        te_code = f"""
    # ── Target encoding (5-fold LOO mean encoding) ──
    from sklearn.model_selection import KFold
    TE_COLS = {repr(te_cols)}
    if y is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for c in TE_COLS:
            if c in X.columns:
                global_mean = float(y.mean())
                encoded = pd.Series(global_mean, index=X.index, dtype=float)
                for tr_idx, val_idx in kf.split(X):
                    means = y.iloc[tr_idx].groupby(X[c].iloc[tr_idx].values).mean()
                    encoded.iloc[val_idx] = X[c].iloc[val_idx].map(means).fillna(global_mean).values
                X[f"{{c}}_te"] = encoded
                X = X.drop(columns=[c])"""

    yj_code = ""
    if yj_cols:
        yj_code = f"""
    # ── Yeo-Johnson transform (skewed with negatives) ──
    from scipy.stats import yeojohnson
    YJ_COLS = {repr(yj_cols)}
    for c in YJ_COLS:
        if c in X.columns:
            clean = X[c].dropna().values
            transformed, _ = yeojohnson(clean)
            X.loc[~X[c].isna(), c] = transformed"""

    return [
        _md(f"""\
# 4. Data Preprocessing & Feature Engineering

{llm_note}

## Pipeline Steps Applied

{steps_md}

## Feature Engineering Decisions (per column)

{fe_md}

| Metric | Value |
|--------|-------|
| Original features | {n_orig} |
| Dropped (bad/irrelevant) | {len(dropped)} |
| Log-transformed | {len(log_cols)} |
| Yeo-Johnson transformed | {len(yj_cols)} |
| Winsorized | {len(winsorized)} |
| Quantile-binned | {len(binned_cols)} |
| Target-encoded | {len(te_cols)} |
| Frequency encoded | {len(freq_cols)} |
| One-hot encoded | {len(ohe_cols)} |
| Cyclical datetime cols | {len(cyclical_cols)} |
| Interaction features created | {n_int} |
| **Final feature count** | **{n_final}** |
"""),
        _code(f"""\
# ═══════════════════════════════════════════════════════════════
# Preprocessing Configuration  (derived from agent analysis)
# ═══════════════════════════════════════════════════════════════

COLS_TO_DROP                  = {repr(dropped)}
INTERACTION_SOURCE_COLS_TO_DROP = {repr(dropped_late)}
LOG_TRANSFORM_COLS = {repr(log_cols)}
YJ_TRANSFORM_COLS  = {repr(yj_cols)}
WINSORIZE_COLS     = {repr(winsorize_cols_all)}
BIN_QUANTILE_COLS  = {repr(binned_cols)}
FREQ_ENCODE_COLS   = {repr(freq_cols)}
OHE_COLS           = {repr(ohe_cols)}
TARGET_ENCODE_COLS = {repr(te_cols)}
CYCLICAL_COLS      = {repr(cyclical_cols)}

print(f"Drop           : {{len(COLS_TO_DROP)}} cols")
print(f"Log-transform  : {{len(LOG_TRANSFORM_COLS)}} cols")
print(f"Yeo-Johnson    : {{len(YJ_TRANSFORM_COLS)}} cols")
print(f"Target-encode  : {{len(TARGET_ENCODE_COLS)}} cols")
print(f"Freq-encode    : {{len(FREQ_ENCODE_COLS)}} cols")
print(f"One-hot        : {{len(OHE_COLS)}} cols")
print(f"Interactions   : {n_int}")
"""),
        _code(f"""\
# ═══════════════════════════════════════════════════════════════
# fit_preprocessor / apply_preprocessor
# ═══════════════════════════════════════════════════════════════

def fit_preprocessor(df, target_col=TARGET):
    \"\"\"Fit on training data. Returns (X, y, transformers).\"\"\"
    df = df.copy()

    # 1. Drop irrelevant columns
    drop = [c for c in COLS_TO_DROP if c in df.columns and c != target_col]
    if drop:
        df = df.drop(columns=drop)

    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")

    # 2. Impute missing values
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    medians = {{c: X[c].median() for c in num_cols if X[c].isnull().any()}}
    modes   = {{c: X[c].mode()[0] for c in cat_cols if X[c].isnull().any()}}
    for c, v in medians.items(): X[c] = X[c].fillna(v)
    for c, v in modes.items():   X[c] = X[c].fillna(v)

    # 3. Winsorise (IQR-based outlier capping)
    iqr_bounds = {{}}
    for c in WINSORIZE_COLS:
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            q1, q3 = X[c].quantile(0.25), X[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_bounds[c] = (lo, hi)
                X[c] = X[c].clip(lower=lo, upper=hi)

    # 4. Log-transform skewed positive columns
    for c in LOG_TRANSFORM_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
{yj_code}
    # 5. Quantile binning for non-linear features
    for c in BIN_QUANTILE_COLS:
        if c in X.columns:
            try:
                X[c] = pd.qcut(X[c], q=5, labels=False, duplicates="drop").astype(float)
            except Exception:
                pass
{interaction_code}
    # Drop interaction source columns now that interactions are created
    _drop_src = [c for c in INTERACTION_SOURCE_COLS_TO_DROP if c in X.columns]
    if _drop_src:
        X = X.drop(columns=_drop_src)

    # 6. Target encoding (if applicable — uses global mean on test set)
    TE_COLS = TARGET_ENCODE_COLS
    if y is not None and TE_COLS:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for c in TE_COLS:
            if c in X.columns:
                gm = float(y.mean())
                enc = pd.Series(gm, index=X.index, dtype=float)
                for tr, va in kf.split(X):
                    m = y.iloc[tr].groupby(X[c].iloc[tr].values).mean()
                    enc.iloc[va] = X[c].iloc[va].map(m).fillna(gm).values
                X[f"{{c}}_te"] = enc
                X = X.drop(columns=[c])

    # 7. Frequency encode high-cardinality categoricals
    freq_maps = {{}}
    for c in FREQ_ENCODE_COLS:
        if c in X.columns:
            fm = X[c].value_counts(normalize=True).to_dict()
            freq_maps[c] = fm
            X[f"{{c}}_freq"] = X[c].map(fm).fillna(0.0)
            X = X.drop(columns=[c])

    # 8. One-hot encode remaining categoricals
    ohe_present = [c for c in OHE_COLS if c in X.columns]
    if ohe_present:
        X = pd.get_dummies(X, columns=ohe_present, drop_first=True)

    # Safety: convert remaining datetime cols to numeric, drop unhandled string cols
    for c in X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist():
        X[c] = X[c].astype("int64").astype(float)
    _residual_obj = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if _residual_obj:
        X = X.drop(columns=_residual_obj)

    feature_names = X.columns.tolist()

    # 9. Scale all numeric features
    scale_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    transformers = dict(
        medians=medians, modes=modes,
        iqr_bounds=iqr_bounds, freq_maps=freq_maps,
        feature_names=feature_names,
        scaler=scaler, scale_cols=scale_cols,
    )
    return X, y, transformers


def apply_preprocessor(df, transformers, target_col=TARGET):
    \"\"\"Apply pre-fitted transformers to new data (test set).\"\"\"
    df = df.copy()
    drop = [c for c in COLS_TO_DROP if c in df.columns and c != target_col]
    if drop:
        df = df.drop(columns=drop)

    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")

    for c, v in transformers["medians"].items():
        if c in X.columns: X[c] = X[c].fillna(v)
    for c, v in transformers["modes"].items():
        if c in X.columns: X[c] = X[c].fillna(v)

    for c, (lo, hi) in transformers["iqr_bounds"].items():
        if c in X.columns: X[c] = X[c].clip(lower=lo, upper=hi)

    for c in LOG_TRANSFORM_COLS:
        if c in X.columns: X[c] = np.log1p(X[c].clip(lower=0))

    # Drop interaction source columns (mirroring fit_preprocessor)
    _drop_src = [c for c in INTERACTION_SOURCE_COLS_TO_DROP if c in X.columns]
    if _drop_src:
        X = X.drop(columns=_drop_src)

    for c, fm in transformers["freq_maps"].items():
        if c in X.columns:
            X[f"{{c}}_freq"] = X[c].map(fm).fillna(0.0)
            X = X.drop(columns=[c])

    ohe_present = [c for c in OHE_COLS if c in X.columns]
    if ohe_present:
        X = pd.get_dummies(X, columns=ohe_present, drop_first=True)

    # Safety: convert remaining datetime cols to numeric, drop unhandled string cols
    for c in X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist():
        X[c] = X[c].astype("int64").astype(float)
    _residual_obj = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if _residual_obj:
        X = X.drop(columns=_residual_obj)

    X = X.reindex(columns=transformers["feature_names"], fill_value=0)

    sc_cols = [c for c in transformers["scale_cols"] if c in X.columns]
    if sc_cols:
        X[sc_cols] = transformers["scaler"].transform(X[sc_cols])

    return X, y


# Apply to full training data
X_full, y_full, transformers = fit_preprocessor(df)
print(f"Preprocessed training shape : {{X_full.shape}}")
print(f"Feature names (first 8)     : {{X_full.columns.tolist()[:8]}}")
X_full.head()
"""),
    ]


def _section_baseline_models(state: dict) -> list:
    problem_type    = state.get("problem_type", "classification")
    baseline_result = state.get("baseline_result", {})
    mlflow_uri      = state.get("mlflow_tracking_uri", "./mlruns")
    mlflow_exp      = state.get("mlflow_experiment_name", "ml_pipeline")
    enable_hpo_mlflow = state.get("enable_hpo_mlflow", True)

    best_name    = baseline_result.get("model", "—")
    best_score   = baseline_result.get("score") or 0.0
    metric       = baseline_result.get("metric", "accuracy" if problem_type == "classification" else "r2")
    reason       = baseline_result.get("selection_reason", "")
    stratify_arg = "stratify=y_full" if problem_type == "classification" else "stratify=None"

    if problem_type == "classification":
        score_fn = """\
def score_model(m, Xtr, Xv, ytr, yv):
    m.fit(Xtr, ytr); p = m.predict(Xv)
    return {"accuracy": round(accuracy_score(yv, p), 4),
            "f1_weighted": round(f1_score(yv, p, average="weighted", zero_division=0), 4)}
SCORE_KEY = "accuracy\""""
        baselines = """\
candidates = [
    ("Dummy (most_frequent)", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ("Logistic Regression",   LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ("Ridge Classifier",      RidgeClassifier()),
    ("Decision Tree",         DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
    ("Random Forest",         RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ("Hist GBM",              HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
]"""
    else:
        score_fn = """\
def score_model(m, Xtr, Xv, ytr, yv):
    m.fit(Xtr, ytr); p = m.predict(Xv)
    return {"r2": round(r2_score(yv, p), 4),
            "mae": round(mean_absolute_error(yv, p), 4)}
SCORE_KEY = "r2\""""
        baselines = """\
candidates = [
    ("Dummy (mean)",      DummyRegressor(strategy="mean")),
    ("Linear Regression", LinearRegression()),
    ("Ridge",             Ridge(random_state=RANDOM_STATE)),
    ("Lasso",             Lasso(random_state=RANDOM_STATE, max_iter=5000)),
    ("Decision Tree",     DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)),
    ("Random Forest",     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ("Hist GBM",          HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
]"""

    return [
        _md("# 5. Baseline Model Benchmarking\n\n"
            "Quick evaluation of simple models to establish a performance floor before tuning. "
            "Every run is logged to MLflow automatically."),
        _code(f"""\
# ── Train / validation split ──────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=RANDOM_STATE, {stratify_arg}
)
print(f"Train : {{X_train.shape}}   Val : {{X_val.shape}}")
"""),
        _code(f"""\
# ── MLflow configuration ──────────────────────────────────────────
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI   = "{mlflow_uri}"
MLFLOW_EXPERIMENT     = "{mlflow_exp}"
# Set to True to log every Optuna trial as a nested MLflow run (section 6)
ENABLE_HPO_MLFLOW     = {str(enable_hpo_mlflow)}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print(f"MLflow tracking URI : {{MLFLOW_TRACKING_URI}}")
print(f"Experiment          : {{MLFLOW_EXPERIMENT}}")
print(f"HPO MLflow logging  : {{ENABLE_HPO_MLFLOW}}")
"""),
        _code(f"""\
# ── Evaluate candidates & log to MLflow ──────────────────────────
{score_fn}

{baselines}

feature_names = transformers["feature_names"]

print("\\nTraining & evaluating baseline models …")
results = []
for name, model in candidates:
    try:
        with mlflow.start_run(run_name=f"baseline_{{name}}") as run:
            s = score_model(model, X_train, X_val, y_train, y_val)
            mlflow.log_param("model_name", name)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("features",   str(feature_names[:30]))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size",   len(X_val))
            mlflow.log_params({{f"metric_{{k}}": v for k, v in s.items()}})
            mlflow.log_metrics(s)
            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception:
                pass
            results.append({{"model": name, "run_id": run.info.run_id, **s}})
            print(f"  {{name:<32}} {{SCORE_KEY}}={{s[SCORE_KEY]:.4f}}  run={{run.info.run_id[:8]}}")
    except Exception as exc:
        print(f"  {{name:<32}} FAILED: {{exc}}")

res_df = pd.DataFrame(results).sort_values(SCORE_KEY, ascending=False).reset_index(drop=True)
print("\\n=== Baseline Results (from MLflow runs) ===")
display(res_df)

# Pull best baseline run back from MLflow
_client = mlflow.tracking.MlflowClient()
_exp    = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
if _exp:
    _runs = _client.search_runs(
        experiment_ids=[_exp.experiment_id],
        filter_string="tags.`mlflow.runName` LIKE 'baseline_%'",
        order_by=[f"metrics.{{SCORE_KEY}} DESC"],
        max_results=1,
    )
    if _runs:
        _best = _runs[0]
        print(f"\\nMLflow best baseline  : {{_best.data.tags.get('mlflow.runName', '?')}}")
        print(f"  {{SCORE_KEY:<12}}: {{_best.data.metrics.get(SCORE_KEY, '?')}}")
        print(f"  run_id       : {{_best.info.run_id}}")

# Visualise
fig, ax = plt.subplots(figsize=(10, max(4, len(res_df) * 0.5)))
colors = ["#22c55e" if i == 0 else "#6366f1" for i in range(len(res_df))]
ax.barh(res_df["model"], res_df[SCORE_KEY], color=colors)
ax.set_xlabel(SCORE_KEY.upper())
ax.set_title(f"Baseline Model Comparison ({{SCORE_KEY.upper()}})", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
"""),
        _md(f"""\
## Baseline Decision

Agent pre-analysis selected **`{best_name}`** as the strongest simple model:
- `{metric}` = `{best_score:.4f}`
- Reasoning: {reason}

All baseline runs are logged to MLflow. Use the chart above (or query MLflow) to confirm or
override this choice before moving to hyperparameter tuning.
"""),
    ]


def _section_hyperparameter_tuning(state: dict) -> list:
    problem_type   = state.get("problem_type", "classification")
    advanced       = state.get("advanced_result", {})
    decision_log   = state.get("decision_log", {})

    best_model_name = advanced.get("model", "HistGradientBoostingClassifier")
    best_params     = advanced.get("best_hyperparameters", {})
    cv_score        = advanced.get("tuning_cv_score") or 0.0
    metric          = advanced.get("metric", "accuracy" if problem_type == "classification" else "r2")

    tuning_strat    = decision_log.get("model_selection", {}).get("tuning_strategy", {})
    cv_folds        = tuning_strat.get("cv_folds", 5)
    n_trials        = min(int(tuning_strat.get("optuna_trials_per_model", 20)), 50)

    scoring = "f1_weighted" if metric == "f1_weighted" else ("accuracy" if problem_type == "classification" else "r2")
    cv_cls  = "StratifiedKFold" if problem_type == "classification" else "KFold"

    is_cls = problem_type == "classification"

    # ── Per-model builder function bodies (indented for notebook code) ──
    if is_cls:
        registry_code = """\
def _build_hist_gbm(trial):
    return HistGradientBoostingClassifier(
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
        max_leaf_nodes   = trial.suggest_int("max_leaf_nodes", 15, 127, step=8),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50, step=5),
        random_state=RANDOM_STATE,
    )

def _build_random_forest(trial):
    return RandomForestClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_extra_trees(trial):
    return ExtraTreesClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_gradient_boost(trial):
    return GradientBoostingClassifier(
        n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=25),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth     = trial.suggest_int("max_depth", 2, 6),
        subsample     = trial.suggest_float("subsample", 0.6, 1.0),
        random_state=RANDOM_STATE,
    )

def _build_adaboost(trial):
    return AdaBoostClassifier(
        n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=25),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        random_state=RANDOM_STATE,
    )

def _build_bagging(trial):
    return BaggingClassifier(
        n_estimators = trial.suggest_int("n_estimators", 10, 100, step=10),
        max_samples  = trial.suggest_float("max_samples", 0.5, 1.0),
        max_features = trial.suggest_float("max_features_frac", 0.5, 1.0),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_decision_tree(trial):
    return DecisionTreeClassifier(
        max_depth        = trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20]),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=RANDOM_STATE,
    )

def _build_logistic(trial):
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE))])

def _build_ridge_cls(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", RidgeClassifier(alpha=alpha))])

def _build_linear_svc(trial):
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LinearSVC(C=C, max_iter=5000, random_state=RANDOM_STATE))])

def _build_svc_rbf(trial):
    C     = trial.suggest_float("C", 1e-2, 100.0, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    return Pipeline([("scaler", StandardScaler()),
                     ("model", SVC(C=C, gamma=gamma, random_state=RANDOM_STATE))])

def _build_knn(trial):
    k      = trial.suggest_int("n_neighbors", 3, 30)
    weight = trial.suggest_categorical("weights", ["uniform", "distance"])
    return Pipeline([("scaler", StandardScaler()),
                     ("model", KNeighborsClassifier(n_neighbors=k, weights=weight))])

def _build_gaussian_nb(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-5, log=True)
    return GaussianNB(var_smoothing=var_smoothing)

def _build_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_sz = trial.suggest_categorical("layer_size", [64, 128, 256])
    alpha    = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", MLPClassifier(
                         hidden_layer_sizes=tuple([layer_sz] * n_layers),
                         alpha=alpha, max_iter=500, random_state=RANDOM_STATE))])

def _build_xgb(trial):
    return XGBClassifier(
        n_estimators       = trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth          = trial.suggest_int("max_depth", 2, 8),
        subsample          = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
    )

def _build_lgbm(trial):
    return LGBMClassifier(
        n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves    = trial.suggest_int("num_leaves", 20, 150),
        subsample     = trial.suggest_float("subsample", 0.6, 1.0),
        random_state=RANDOM_STATE, verbosity=-1,
    )

MODEL_CANDIDATES = [
    ("HistGradientBoosting", _build_hist_gbm),
    ("RandomForest",         _build_random_forest),
    ("ExtraTrees",           _build_extra_trees),
    ("GradientBoosting",     _build_gradient_boost),
    ("AdaBoost",             _build_adaboost),
    ("BaggingClassifier",    _build_bagging),
    ("DecisionTree",         _build_decision_tree),
    ("LogisticRegression",   _build_logistic),
    ("RidgeClassifier",      _build_ridge_cls),
    ("LinearSVC",            _build_linear_svc),
    ("SVC-RBF",              _build_svc_rbf),
    ("KNN",                  _build_knn),
    ("GaussianNB",           _build_gaussian_nb),
    ("MLP",                  _build_mlp),
]
if HAS_XGB:
    MODEL_CANDIDATES.append(("XGBoost",  _build_xgb))
if HAS_LGBM:
    MODEL_CANDIDATES.append(("LightGBM", _build_lgbm))"""
    else:
        registry_code = """\
def _build_hist_gbm(trial):
    return HistGradientBoostingRegressor(
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
        max_leaf_nodes   = trial.suggest_int("max_leaf_nodes", 15, 127, step=8),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50, step=5),
        random_state=RANDOM_STATE,
    )

def _build_random_forest(trial):
    return RandomForestRegressor(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_extra_trees(trial):
    return ExtraTreesRegressor(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth        = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_gradient_boost(trial):
    return GradientBoostingRegressor(
        n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=25),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth     = trial.suggest_int("max_depth", 2, 6),
        subsample     = trial.suggest_float("subsample", 0.6, 1.0),
        random_state=RANDOM_STATE,
    )

def _build_adaboost(trial):
    return AdaBoostRegressor(
        n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=25),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        random_state=RANDOM_STATE,
    )

def _build_bagging(trial):
    return BaggingRegressor(
        n_estimators = trial.suggest_int("n_estimators", 10, 100, step=10),
        max_samples  = trial.suggest_float("max_samples", 0.5, 1.0),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

def _build_decision_tree(trial):
    return DecisionTreeRegressor(
        max_depth        = trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20]),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=RANDOM_STATE,
    )

def _build_linear(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", Ridge(alpha=alpha, random_state=RANDOM_STATE))])

def _build_lasso(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", Lasso(alpha=alpha, max_iter=5000, random_state=RANDOM_STATE))])

def _build_elasticnet(trial):
    alpha   = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    l1_ratio= trial.suggest_float("l1_ratio", 0.1, 0.9)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=RANDOM_STATE))])

def _build_svr_rbf(trial):
    C   = trial.suggest_float("C", 1e-2, 100.0, log=True)
    eps = trial.suggest_float("epsilon", 0.01, 1.0)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", SVR(C=C, epsilon=eps))])

def _build_knn(trial):
    k      = trial.suggest_int("n_neighbors", 3, 30)
    weight = trial.suggest_categorical("weights", ["uniform", "distance"])
    return Pipeline([("scaler", StandardScaler()),
                     ("model", KNeighborsRegressor(n_neighbors=k, weights=weight))])

def _build_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_sz = trial.suggest_categorical("layer_size", [64, 128, 256])
    alpha    = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    return Pipeline([("scaler", StandardScaler()),
                     ("model", MLPRegressor(
                         hidden_layer_sizes=tuple([layer_sz] * n_layers),
                         alpha=alpha, max_iter=500, random_state=RANDOM_STATE))])

def _build_xgb(trial):
    return XGBRegressor(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth        = trial.suggest_int("max_depth", 2, 8),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=RANDOM_STATE, verbosity=0,
    )

def _build_lgbm(trial):
    return LGBMRegressor(
        n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves    = trial.suggest_int("num_leaves", 20, 150),
        subsample     = trial.suggest_float("subsample", 0.6, 1.0),
        random_state=RANDOM_STATE, verbosity=-1,
    )

MODEL_CANDIDATES = [
    ("HistGradientBoosting", _build_hist_gbm),
    ("RandomForest",         _build_random_forest),
    ("ExtraTrees",           _build_extra_trees),
    ("GradientBoosting",     _build_gradient_boost),
    ("AdaBoost",             _build_adaboost),
    ("BaggingRegressor",     _build_bagging),
    ("DecisionTree",         _build_decision_tree),
    ("Ridge",                _build_linear),
    ("Lasso",                _build_lasso),
    ("ElasticNet",           _build_elasticnet),
    ("SVR-RBF",              _build_svr_rbf),
    ("KNN",                  _build_knn),
    ("MLP",                  _build_mlp),
]
if HAS_XGB:
    MODEL_CANDIDATES.append(("XGBoost",  _build_xgb))
if HAS_LGBM:
    MODEL_CANDIDATES.append(("LightGBM", _build_lgbm))"""

    return [
        _md("# 6. Hyperparameter Tuning\n\n"
            "Bayesian optimisation with Optuna (TPE sampler) over **all** candidate model families. "
            "Each model gets its own study. Compare results at the end and pick the best."),
        _code(f"""\
# ── Tuning configuration ─────────────────────────────────────────
SCORING   = {repr(scoring)}   # metric to maximise
N_TRIALS  = {n_trials}        # trials per model — increase for better results
CV_FOLDS  = {cv_folds}
cv_split  = {cv_cls}(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── Helper: wrap a builder function into an Optuna objective ──────
def make_objective(build_fn, parent_run_id=None):
    def objective(trial):
        model  = build_fn(trial)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv_split, scoring=SCORING, n_jobs=-1,
                                 error_score="raise")
        cv_mean = float(scores.mean())
        if ENABLE_HPO_MLFLOW and parent_run_id:
            with mlflow.start_run(run_name=f"trial_{{trial.number}}", nested=True,
                                  parent_run_id=parent_run_id):
                mlflow.log_params(trial.params)
                mlflow.log_metric(SCORING, cv_mean)
                mlflow.log_metric("cv_std", float(scores.std()))
        return cv_mean
    return objective

# ── All model builder functions + candidate list ──────────────────
{registry_code}
"""),
        _code(f"""\
# ── Run one Optuna study per model ───────────────────────────────
tuning_results = []

if HAS_OPTUNA:
    for model_name, build_fn in MODEL_CANDIDATES:
        print(f"  Tuning {{model_name:<28}} ({{N_TRIALS}} trials) ...", end=" ", flush=True)
        try:
            _parent_run_id = None
            _parent_ctx    = None

            if ENABLE_HPO_MLFLOW:
                _parent_ctx = mlflow.start_run(run_name=f"hpo_{{model_name}}")
                _parent_run = _parent_ctx.__enter__()
                _parent_run_id = _parent_run.info.run_id
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_trials",   N_TRIALS)
                mlflow.log_param("cv_folds",   CV_FOLDS)
                mlflow.log_param("scoring",    SCORING)

            study = optuna.create_study(
                study_name=model_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            study.optimize(make_objective(build_fn, parent_run_id=_parent_run_id),
                           n_trials=N_TRIALS, show_progress_bar=False)

            best_score = round(study.best_value, 4)
            tuning_results.append({{
                "model":       model_name,
                "best_score":  best_score,
                "n_trials":    len(study.trials),
                "best_params": study.best_params,
                "run_id":      _parent_run_id or "",
            }})

            if ENABLE_HPO_MLFLOW and _parent_ctx is not None:
                mlflow.log_metric(f"best_{{SCORING}}", best_score)
                mlflow.log_params({{f"best_{{k}}": v for k, v in study.best_params.items()}})
                # Rebuild and log best model artifact
                try:
                    _best_model = build_fn(study.best_trial)
                    _best_model.fit(X_train, y_train)
                    mlflow.sklearn.log_model(_best_model, artifact_path="best_model")
                except Exception:
                    pass
                _parent_ctx.__exit__(None, None, None)

            print(f"{{SCORING}} = {{study.best_value:.4f}}"
                  + (f"  run={{_parent_run_id[:8]}}" if _parent_run_id else ""))
        except Exception as exc:
            print(f"FAILED — {{exc}}")
            try:
                if ENABLE_HPO_MLFLOW and _parent_ctx is not None:
                    _parent_ctx.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass

    print(f"\\nTuning complete. {{len(tuning_results)}} models evaluated.")
else:
    print("Optuna not available — install with: pip install optuna")
    tuning_results = []
"""),
        _code(f"""\
# ── Compare all models ────────────────────────────────────────────
if tuning_results:
    tuning_df = pd.DataFrame(tuning_results).sort_values("best_score", ascending=False).reset_index(drop=True)
    display(tuning_df[["model", "best_score", "n_trials", "run_id"]])

    fig, ax = plt.subplots(figsize=(12, max(5, len(tuning_df) * 0.5)))
    colors = ["#22c55e" if i == 0 else "#6366f1" for i in range(len(tuning_df))]
    ax.barh(tuning_df["model"], tuning_df["best_score"], color=colors)
    ax.set_xlabel(SCORING.upper())
    ax.set_title(f"Hyperparameter Tuning — All Models ({{SCORING.upper()}})", fontsize=13, fontweight="bold")
    ax.axvline(tuning_df["best_score"].max(), color="green", ls="--", alpha=0.5, label="Best")
    ax.legend(); plt.tight_layout(); plt.show()

    BEST_MODEL_NAME = tuning_df.iloc[0]["model"]
    BEST_PARAMS     = tuning_df.iloc[0]["best_params"]
    print(f"\\nBest model : {{BEST_MODEL_NAME}}")
    print(f"Best {{SCORING}}: {{tuning_df.iloc[0]['best_score']:.4f}}")
    print(f"Best params: {{BEST_PARAMS}}")
else:
    print("No tuning results — install optuna: pip install optuna")

# ── Pull best HPO run from MLflow (if logged) ─────────────────────
if ENABLE_HPO_MLFLOW:
    _client = mlflow.tracking.MlflowClient()
    _exp    = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if _exp:
        _hpo_runs = _client.search_runs(
            experiment_ids=[_exp.experiment_id],
            filter_string="tags.`mlflow.runName` LIKE 'hpo_%'",
            order_by=[f"metrics.best_{{SCORING}} DESC"],
            max_results=1,
        )
        if _hpo_runs:
            _best_hpo = _hpo_runs[0]
            print(f"\\nMLflow best HPO run   : {{_best_hpo.data.tags.get('mlflow.runName', '?')}}")
            print(f"  best_{{SCORING:<8}}  : {{_best_hpo.data.metrics.get(f'best_{{SCORING}}', '?')}}")
            print(f"  run_id         : {{_best_hpo.info.run_id}}")
            print(f"  To inspect     : mlflow ui --backend-store-uri {{MLFLOW_TRACKING_URI}}")
"""),
        _md(f"""\
## Best Model Decision

Agent pre-analysis selected **`{best_model_name}`** (CV `{metric}` = `{cv_score:.4f}`).

After running your own multi-model tuning above, `BEST_MODEL_NAME` and `BEST_PARAMS` are set
to whichever model topped the comparison chart. Update them manually if you prefer a different
model (e.g. for interpretability or inference speed).
"""),
    ]


def _section_best_model(state: dict) -> list:
    problem_type = state.get("problem_type", "classification")
    advanced     = state.get("advanced_result", {})
    baseline     = state.get("baseline_result", {})

    # Use advanced if available, else fall back to baseline
    result = advanced if advanced else baseline
    model_id     = result.get("model_id", "")
    model_name   = result.get("model", "HistGradientBoosting")
    best_params  = result.get("best_hyperparameters", result.get("best_params", {}))
    metric       = result.get("metric", "accuracy" if problem_type == "classification" else "r2")
    score        = result.get("score") or 0.0

    info = _get_model_info(model_id, problem_type, best_params)
    best_params_json = json.dumps(best_params, indent=4)

    scale_wrap_fit   = ""
    scale_wrap_apply = "best_model"
    if info["needs_scaling"]:
        scale_wrap_fit   = (
            "\nbest_model = Pipeline([('scaler', StandardScaler()), ('model', best_model)])"
        )
        scale_wrap_apply = "best_model"  # pipeline wraps both

    return [
        _md(f"""\
# 7. Best Model — Full Evaluation

| | |
|---|---|
| **Model** | `{model_name}` |
| **Primary metric ({metric})** | `{score:.4f}` |
| **sklearn class** | `{info['class_name']}` |

Training on the 80 % train split and evaluating on the 20 % validation split.
"""),
        _code(f"""\
# ── Instantiate best model ────────────────────────────────────────
{info['import_line']}

best_params_final = BEST_PARAMS if "BEST_PARAMS" in dir() else {best_params_json}

# Strip pipeline-prefix keys and known-fixed params
_SKIP = {{"random_state", "n_jobs", "max_iter", "early_stopping", "eval_metric"}}
clean_p = {{k.replace("model__", ""): v
           for k, v in best_params_final.items()
           if k.replace("model__", "") not in _SKIP}}

try:
    best_model = {info['class_name']}(
        **clean_p,
        random_state=RANDOM_STATE,
        {"n_jobs=-1," if info["class_name"] in {"RandomForestClassifier","ExtraTreesClassifier","BaggingClassifier","RandomForestRegressor","ExtraTreesRegressor","BaggingRegressor"} else ""}
    )
except TypeError:
    best_model = {info['class_name']}(random_state=RANDOM_STATE)
{scale_wrap_fit}
# Fit and evaluate
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)

if PROBLEM_TYPE == "classification":
    val_acc = accuracy_score(y_val, y_pred)
    val_f1  = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    print(f"Validation accuracy    : {{val_acc:.4f}}")
    print(f"Validation F1 weighted : {{val_f1:.4f}}")
    print("\\n" + classification_report(y_val, y_pred))
else:
    val_r2   = r2_score(y_val, y_pred)
    val_mae  = mean_absolute_error(y_val, y_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation R²   : {{val_r2:.4f}}")
    print(f"Validation MAE  : {{val_mae:.4f}}")
    print(f"Validation RMSE : {{val_rmse:.4f}}")
"""),
        _code(f"""\
# ── Evaluation plots ─────────────────────────────────────────────
if PROBLEM_TYPE == "classification":
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_val, y_pred, alpha=0.4, color="#6366f1", edgecolors="none")
    lo = min(y_val.min(), y_pred.min()); hi = max(y_val.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted", fontweight="bold"); axes[0].legend()

    residuals = y_val - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, color="#f59e0b", edgecolors="none")
    axes[1].axhline(0, color="r", ls="--", lw=2)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot", fontweight="bold")
    plt.suptitle(f"Model Evaluation — R² = {{val_r2:.4f}}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# ── Feature importance (if available) ────────────────────────────
inner = getattr(best_model, "named_steps", {{"model": best_model}}).get("model", best_model)
feat_names = X_full.columns.tolist()

if hasattr(inner, "feature_importances_"):
    imp = pd.Series(inner.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, max(5, len(imp) * 0.4)))
    imp.sort_values().plot(kind="barh", ax=ax, color="#6366f1", edgecolor="white")
    ax.set_title("Top Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance"); plt.tight_layout(); plt.show()

elif hasattr(inner, "coef_"):
    coef = np.abs(inner.coef_).mean(axis=0) if inner.coef_.ndim > 1 else np.abs(inner.coef_)
    n = min(len(feat_names), len(coef))
    cs = pd.Series(coef[:n], index=feat_names[:n]).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, max(5, len(cs) * 0.4)))
    cs.sort_values().plot(kind="barh", ax=ax, color="#6366f1", edgecolor="white")
    ax.set_title("Feature Coefficients (Absolute)", fontsize=13, fontweight="bold")
    ax.set_xlabel("|Coefficient|"); plt.tight_layout(); plt.show()
"""),
    ]


def _section_test_pipeline(state: dict) -> list:
    target       = state.get("target_column", "target")
    dataset_path = state.get("dataset_path", "data.csv")
    problem_type = state.get("problem_type", "classification")
    advanced     = state.get("advanced_result", {})
    baseline     = state.get("baseline_result", {})

    result   = advanced if advanced else baseline
    model_id = result.get("model_id", "")
    info     = _get_model_info(model_id, problem_type, result.get("best_hyperparameters", {}))

    scale_wrap = (
        "Pipeline([('scaler', StandardScaler()), ('final_model', final_estimator)])"
        if info["needs_scaling"] else "final_estimator"
    )

    return [
        _md("# 8. Test Data Pipeline & Final Predictions\n\n"
            "Retrain on the **full** training set, apply the same preprocessing to test data, "
            "and generate predictions."),
        _code(f"""\
# ── Load & preprocess test data ───────────────────────────────────
if os.path.exists(TEST_PATH):
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test set shape: {{test_df.shape}}")
    X_test, _ = apply_preprocessor(test_df, transformers, target_col=TARGET)
    print(f"Preprocessed test: {{X_test.shape}}")
    HAS_TEST = True
else:
    print(f"Test file not found at: {{TEST_PATH}}")
    print("Using validation split as demonstration.")
    test_df = df.iloc[int(0.8 * len(df)):].reset_index(drop=True)
    X_test, _ = apply_preprocessor(test_df, transformers, target_col=TARGET)
    HAS_TEST = False

# ── Retrain best model on ALL training data ───────────────────────
{info['import_line']}

_SKIP = {{"random_state", "n_jobs", "max_iter", "early_stopping", "eval_metric"}}
clean_final = {{k.replace("model__", ""): v
               for k, v in (BEST_PARAMS if "BEST_PARAMS" in dir() else {{}}).items()
               if k.replace("model__", "") not in _SKIP}}

try:
    final_estimator = {info['class_name']}(
        **clean_final,
        random_state=RANDOM_STATE,
        {"n_jobs=-1," if info["class_name"] in {"RandomForestClassifier","ExtraTreesClassifier","BaggingClassifier","RandomForestRegressor","ExtraTreesRegressor","BaggingRegressor"} else ""}
    )
except TypeError:
    final_estimator = {info['class_name']}(random_state=RANDOM_STATE)

final_pipeline = {scale_wrap}
final_pipeline.fit(X_full, y_full)

final_preds = final_pipeline.predict(X_test)
print(f"Predictions shape: {{final_preds.shape}}")
print(f"Sample            : {{final_preds[:10]}}")
"""),
        _code(f"""\
# ── Build submission CSV ─────────────────────────────────────────
id_col = next(
    (c for c in ["id", "ID", "Id", "row_id", "PassengerId"] if c in test_df.columns),
    None,
)

if id_col:
    submission = pd.DataFrame({{id_col: test_df[id_col], TARGET: final_preds}})
else:
    submission = pd.DataFrame({{"id": range(len(final_preds)), TARGET: final_preds}})

submission.to_csv("submission.csv", index=False)
print("Saved  : submission.csv")
print(f"Shape  : {{submission.shape}}")
display(submission.head(10))
"""),
    ]


def _section_results(state: dict) -> list:
    problem_type = state.get("problem_type", "classification")
    advanced     = state.get("advanced_result", {})
    baseline     = state.get("baseline_result", {})

    result     = advanced if advanced else baseline
    model_name = result.get("model", "—")
    metric     = result.get("metric", "accuracy" if problem_type == "classification" else "r2")
    score      = result.get("score") or 0.0
    base_score = baseline.get("score") or 0.0
    gain       = score - base_score if advanced else 0.0

    return [
        _md(f"""\
# 9. Results & Next Steps

| | |
|---|---|
| **Best model** | `{model_name}` |
| **Primary metric ({metric})** | `{score:.4f}` |
| **Baseline score** | `{base_score:.4f}` |
| **Improvement over baseline** | `{gain:+.4f}` |

## Suggestions for Further Improvement

- **Feature engineering**: polynomial interactions, domain-specific features
- **Ensemble / stacking**: blend top-3 models from the tuning comparison
- **Cross-validation**: use full CV (5-fold) for final score estimates
- **Calibration** (classification): `CalibratedClassifierCV` for probability outputs
- **Domain knowledge**: add external data sources
"""),
        _code(f"""\
# ── Final summary ─────────────────────────────────────────────────
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)
print(f"  Best model   : {model_name}")
print(f"  Metric       : {metric}")
print(f"  Agent score  : {score:.4f}")

if PROBLEM_TYPE == "classification":
    print(f"  Val accuracy : {{accuracy_score(y_val, y_pred):.4f}}")
    print(f"  Val F1-w     : {{f1_score(y_val, y_pred, average='weighted', zero_division=0):.4f}}")
else:
    print(f"  Val R²       : {{r2_score(y_val, y_pred):.4f}}")
    print(f"  Val MAE      : {{mean_absolute_error(y_val, y_pred):.4f}}")
    print(f"  Val RMSE     : {{np.sqrt(mean_squared_error(y_val, y_pred)):.4f}}")

print("=" * 60)
print("Notebook complete!  submission.csv is ready.")
"""),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def notebook_agent(state: dict) -> dict:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

    cells: list = []
    cells += _section_title(state)
    cells += _section_mlflow_info(state)
    cells += _section_setup(state)
    cells += _section_data_loading(state)
    cells += _section_eda(state)
    cells += _section_statistical_analysis(state)
    cells += _section_preprocessing(state)
    cells += _section_baseline_models(state)
    cells += _section_hyperparameter_tuning(state)
    cells += _section_best_model(state)
    cells += _section_test_pipeline(state)
    cells += _section_results(state)

    nb.cells = cells

    output_path = "output_notebook.ipynb"
    nbf.write(nb, output_path)
    print(f"Notebook generated at: {output_path}  ({len(cells)} cells)")

    state["notebook_json"] = nb
    return state
