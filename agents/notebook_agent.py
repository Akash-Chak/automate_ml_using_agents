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
    dropped = list(preprocessing_report.get("dropped_columns", []))
    transformed = preprocessing_report.get("transformed_columns", [])
    encoded = preprocessing_report.get("encoded_columns", {"frequency": [], "one_hot": []})

    log_cols = [c.split(":")[0] for c in transformed if ":log1p" in c]
    winsorized_cols = [c.split(":")[0] for c in transformed if ":winsorized" in c]
    freq_cols = list(encoded.get("frequency", []))
    ohe_cols = list(encoded.get("one_hot", []))

    return {
        "dropped": dropped,
        "log_cols": log_cols,
        "winsorized_cols": winsorized_cols,
        "freq_cols": freq_cols,
        "ohe_cols": ohe_cols,
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

    lines = ["## MLflow Experiment Tracking", ""]
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


def _section_setup() -> list:
    return [
        _md("## Setup\n\nImport all libraries needed for the full ML pipeline."),
        _code("""\
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier,
    LinearRegression,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,    RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier,      ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier,        AdaBoostRegressor,
    BaggingClassifier,         BaggingRegressor,
)
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
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
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

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
        _md("## 1. Data Loading & Initial Exploration"),
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
    stats_report = state.get("stats_report", {})

    insights = eda_report.get("insights", [])[:8]
    insights_md = "\n".join(f"- {i}" for i in insights) if insights else "- No insights available."

    top_feats = _top_features(stats_report, top_n=6)
    top_feats_repr = repr(top_feats)

    target_plot = ""
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
        _md(f"""\
## 2. Exploratory Data Analysis

### Key insights from agent analysis

{insights_md}
"""),
        _code(f"""\
# ── Target distribution ──────────────────────────────────────────
{target_plot}
"""),
        _code(f"""\
# ── Top significant feature distributions ────────────────────────
TOP_FEATURES = {top_feats_repr}
TOP_FEATURES = [f for f in TOP_FEATURES if f in df.columns][:6]

if TOP_FEATURES:
    n_cols = 3
    n_rows = (len(TOP_FEATURES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(TOP_FEATURES):
        ax = axes_flat[i]
        if pd.api.types.is_numeric_dtype(df[col]):
            ax.hist(df[col].dropna(), bins=30, density=True,
                    alpha=0.7, color="#6366f1", edgecolor="white")
            ax.set_title(col, fontweight="bold", fontsize=11)
            ax.set_xlabel(col); ax.set_ylabel("Density")
        else:
            vc = df[col].value_counts().head(10)
            ax.bar(vc.index.astype(str), vc.values, color="#6366f1", alpha=0.8)
            ax.set_title(col, fontweight="bold", fontsize=11)
            ax.set_xlabel(col); ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

    for j in range(len(TOP_FEATURES), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Top Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
"""),
        _code(f"""\
# ── Correlation heatmap ───────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols) >= 2:
    target_in_num = TARGET in num_cols
    if target_in_num:
        tc_abs = df[num_cols].corr()[TARGET].abs().sort_values(ascending=False)
        plot_cols = tc_abs.head(20).index.tolist()
    else:
        plot_cols = num_cols[:20]

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
        tc = df[num_cols].corr()[TARGET].drop(TARGET).abs() \\
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
    ]


def _section_statistical_analysis(state: dict) -> list:
    stats_report = state.get("stats_report", {})

    significant   = stats_report.get("significant_after_fdr", [])
    insignificant = stats_report.get("insignificant", [])
    ranked        = stats_report.get("ranked_features", [])[:20]
    large_effect  = stats_report.get("large_effect_features", [])
    medium_effect = stats_report.get("medium_effect_features", [])

    sig_bullets = "\n".join(f"  - `{f}`" for f in significant[:15]) or "  - None identified"
    large_str   = ", ".join(f"`{f}`" for f in large_effect[:8]) or "None"
    medium_str  = ", ".join(f"`{f}`" for f in medium_effect[:8]) or "None"

    ranked_json = json.dumps(ranked, indent=2)
    large_json  = json.dumps(large_effect)
    medium_json = json.dumps(medium_effect)

    return [
        _md(f"""\
## 3. Statistical Feature Significance

Statistical tests (t-test / Mann-Whitney / ANOVA / Kruskal-Wallis / Chi-square / Correlation)
were run on every feature. Benjamini-Hochberg FDR correction was applied.

**Significant ({len(significant)} features, p < 0.05 after FDR):**

{sig_bullets}

**Large effect size:** {large_str}
**Medium effect size:** {medium_str}
**Insignificant / deprioritised:** {len(insignificant)} features
"""),
        _code(f"""\
# ── Feature significance visualisation ───────────────────────────
ranked_features = {ranked_json}
large_effect    = {large_json}
medium_effect   = {medium_json}

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
        from matplotlib.patches import Patch
        axes[0].legend(handles=[
            Patch(color="#22c55e", label="Large"),
            Patch(color="#f59e0b", label="Medium"),
            Patch(color="#64748b", label="Small"),
        ])

    if "p_value" in rdf.columns:
        rdf["neg_log_p"] = -np.log10(rdf["p_value"].clip(lower=1e-300))
        rdf_s2 = rdf.sort_values("neg_log_p", ascending=True).tail(15)
        c2 = ["#22c55e" if p < 0.01 else "#f59e0b" if p < 0.05 else "#64748b"
               for p in rdf_s2["p_value"]]
        axes[1].barh(rdf_s2["feature"].astype(str), rdf_s2["neg_log_p"], color=c2)
        axes[1].axvline(-np.log10(0.05), color="orange", ls="--", label="p = 0.05")
        axes[1].set_title("-log₁₀(p-value)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("-log₁₀(p)")
        axes[1].legend()

    plt.suptitle("Statistical Feature Significance", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("No ranked-feature data available.")
"""),
    ]


def _section_preprocessing(state: dict) -> list:
    preprocessing_report = state.get("preprocessing_report", {})
    target = state.get("target_column", "target")

    cfg = _parse_preprocessing(preprocessing_report)
    dropped       = cfg["dropped"]
    log_cols      = cfg["log_cols"]
    winsorized    = cfg["winsorized_cols"]
    freq_cols     = cfg["freq_cols"]
    ohe_cols      = cfg["ohe_cols"]

    steps   = preprocessing_report.get("steps", [])
    n_orig  = preprocessing_report.get("original_feature_count", "?")
    n_final = preprocessing_report.get("selected_feature_count", "?")

    steps_md = "\n".join(f"  - {s.replace('_', ' ').title()}" for s in steps) or "  - Standard pipeline"

    winsorize_cols_all = list(set(winsorized + log_cols))

    return [
        _md(f"""\
## 4. Data Preprocessing

### Agent preprocessing decisions

The preprocessing agent analysed the data and applied:

{steps_md}

| Metric | Value |
|--------|-------|
| Original features | {n_orig} |
| Dropped (bad/irrelevant) | {len(dropped)} |
| Log-transformed | {len(log_cols)} |
| Frequency encoded | {len(freq_cols)} |
| One-hot encoded | {len(ohe_cols)} |
| **Final feature count** | **{n_final}** |
"""),
        _code(f"""\
# ═══════════════════════════════════════════════════════════════
# Preprocessing Configuration  (derived from agent analysis)
# ═══════════════════════════════════════════════════════════════

COLS_TO_DROP      = {repr(dropped)}
LOG_TRANSFORM_COLS= {repr(log_cols)}
WINSORIZE_COLS    = {repr(winsorize_cols_all)}   # capped before log-transform
FREQ_ENCODE_COLS  = {repr(freq_cols)}
OHE_COLS          = {repr(ohe_cols)}

print(f"Drop        : {{len(COLS_TO_DROP)}} cols")
print(f"Log-transform : {{len(LOG_TRANSFORM_COLS)}} cols")
print(f"Freq-encode  : {{len(FREQ_ENCODE_COLS)}} cols")
print(f"One-hot      : {{len(OHE_COLS)}} cols")
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

    # 4. Log-transform skewed columns
    for c in LOG_TRANSFORM_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))

    # 5. Frequency encode high-cardinality categoricals
    freq_maps = {{}}
    for c in FREQ_ENCODE_COLS:
        if c in X.columns:
            fm = X[c].value_counts(normalize=True).to_dict()
            freq_maps[c] = fm
            X[f"{{c}}_freq"] = X[c].map(fm).fillna(0.0)
            X = X.drop(columns=[c])

    # 6. One-hot encode remaining categoricals
    ohe_present = [c for c in OHE_COLS if c in X.columns]
    if ohe_present:
        X = pd.get_dummies(X, columns=ohe_present, drop_first=True)

    feature_names = X.columns.tolist()

    # 7. Scale all numeric features
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

    for c, fm in transformers["freq_maps"].items():
        if c in X.columns:
            X[f"{{c}}_freq"] = X[c].map(fm).fillna(0.0)
            X = X.drop(columns=[c])

    ohe_present = [c for c in OHE_COLS if c in X.columns]
    if ohe_present:
        X = pd.get_dummies(X, columns=ohe_present, drop_first=True)

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

    best_name    = baseline_result.get("model", "—")
    best_score   = baseline_result.get("score") or 0.0
    metric       = baseline_result.get("metric", "accuracy" if problem_type == "classification" else "r2")
    reason       = baseline_result.get("selection_reason", "")
    candidates   = baseline_result.get("candidate_results", [])
    cand_json    = json.dumps(candidates[:10], indent=2)

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
        _md(f"""\
## 5. Baseline Model Benchmarking

Quick evaluation of simple models to bound performance before tuning.

**Agent baseline result:**
- Best model: `{best_name}`
- `{metric}` = `{best_score:.4f}`
- Reasoning: {reason}
"""),
        _code(f"""\
# ── Train / validation split ──────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=RANDOM_STATE, {stratify_arg}
)
print(f"Train : {{X_train.shape}}   Val : {{X_val.shape}}")

# Agent's candidate results for reference
AGENT_BASELINE = {cand_json}
"""),
        _code(f"""\
# ── Evaluate candidates ───────────────────────────────────────────
{score_fn}

{baselines}

print("\\nTraining & evaluating baseline models …")
results = []
for name, model in candidates:
    try:
        s = score_model(model, X_train, X_val, y_train, y_val)
        results.append({{"model": name, **s}})
        print(f"  {{name:<32}} {{SCORE_KEY}}={{s[SCORE_KEY]:.4f}}")
    except Exception as exc:
        print(f"  {{name:<32}} FAILED: {{exc}}")

res_df = pd.DataFrame(results).sort_values(SCORE_KEY, ascending=False)
print("\\n=== Baseline Results ===")
display(res_df)

# Visualise
fig, ax = plt.subplots(figsize=(10, max(4, len(res_df) * 0.5)))
colors = ["#22c55e" if i == 0 else "#6366f1" for i in range(len(res_df))]
ax.barh(res_df["model"], res_df[SCORE_KEY], color=colors)
ax.set_xlabel(SCORE_KEY.upper())
ax.set_title(f"Baseline Model Comparison ({{SCORE_KEY.upper()}})", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    ]


def _section_hyperparameter_tuning(state: dict) -> list:
    problem_type   = state.get("problem_type", "classification")
    advanced       = state.get("advanced_result", {})
    decision_log   = state.get("decision_log", {})

    best_model_id   = advanced.get("model_id", "")
    best_model_name = advanced.get("model", "HistGradientBoostingClassifier")
    best_params     = advanced.get("best_hyperparameters", {})
    cv_score        = advanced.get("tuning_cv_score") or 0.0
    metric          = advanced.get("metric", "accuracy" if problem_type == "classification" else "r2")
    cand_results    = advanced.get("candidate_results", [])
    cand_json       = json.dumps(cand_results[:10], indent=2)

    tuning_strat    = decision_log.get("model_selection", {}).get("tuning_strategy", {})
    cv_folds        = tuning_strat.get("cv_folds", 5)
    n_trials        = min(int(tuning_strat.get("optuna_trials_per_model", 30)), 50)

    suggest_block = _optuna_suggest_block(best_model_id, problem_type)

    scoring = "f1_weighted" if metric == "f1_weighted" else ("accuracy" if problem_type == "classification" else "r2")
    cv_cls  = "StratifiedKFold" if problem_type == "classification" else "KFold"

    best_params_json = json.dumps(best_params, indent=4)

    return [
        _md(f"""\
## 6. Hyperparameter Tuning

Bayesian optimisation with Optuna (TPE sampler) over the most promising model families.

**Agent best result:**
- Model: `{best_model_name}`
- CV `{metric}` = `{cv_score:.4f}`
- Best params: `{json.dumps(best_params)}`
"""),
        _code(f"""\
# ── Agent tuning results (reference) ─────────────────────────────
AGENT_TUNED   = {cand_json}
AGENT_BEST_PARAMS = {best_params_json}

print("Agent best model  :", {repr(best_model_name)})
print("Agent best params :", AGENT_BEST_PARAMS)
print(f"Agent CV {metric}    : {cv_score:.4f}")

# ── Run Optuna study (or skip and use AGENT_BEST_PARAMS) ──────────
N_TRIALS  = {n_trials}
CV_FOLDS  = {cv_folds}
cv_split  = {cv_cls}(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def optuna_objective(trial):
{suggest_block}
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv_split, scoring={repr(scoring)}, n_jobs=-1,
    )
    return float(scores.mean())

if HAS_OPTUNA:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\\nOptuna best {scoring}: {{study.best_value:.4f}}")
    print(f"Optuna best params : {{study.best_params}}")
    BEST_PARAMS = {{**AGENT_BEST_PARAMS, **study.best_params}}
else:
    print("Optuna not available — using agent's pre-tuned parameters.")
    BEST_PARAMS = AGENT_BEST_PARAMS

print(f"\\nFinal hyperparameters: {{BEST_PARAMS}}")
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
## 7. Best Model — Full Evaluation

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
        _md("## 8. Test Data Pipeline & Final Predictions\n\n"
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
## 9. Results & Next Steps

| | |
|---|---|
| **Best model** | `{model_name}` |
| **Primary metric ({metric})** | `{score:.4f}` |
| **Baseline score** | `{base_score:.4f}` |
| **Improvement over baseline** | `{gain:+.4f}` |

### Suggestions for further improvement

- **Feature engineering**: polynomial interactions, domain-specific features
- **Ensemble / stacking**: blend top-3 models
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
    cells += _section_setup()
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
