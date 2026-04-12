# agents/eda_agent.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats


# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "outputs/eda"


def _save(fig, name: str, plots: list):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    plots.append(path)


# ── Sub-analyses ──────────────────────────────────────────────────────────────

def _analyse_numerical(df: pd.DataFrame, col: str, target: str,
                       problem_type: str, plots: list, insights: list) -> dict:
    series = df[col].dropna()

    mean_val   = float(series.mean())
    median_val = float(series.median())
    std_val    = float(series.std())
    skew_val   = float(series.skew())
    kurt_val   = float(series.kurt())
    q1, q3     = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr        = q3 - q1
    outlier_pct = float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() * 100)

    # Normality test (only if sample is manageable)
    if len(series) <= 5000:
        _, p_normal = stats.shapiro(series.sample(min(len(series), 500), random_state=42))
    else:
        _, p_normal = stats.kstest(series, "norm", args=(mean_val, std_val))
    is_normal = p_normal > 0.05

    # Insight flags
    if abs(skew_val) > 1:
        insights.append(f"⚠️  '{col}' is heavily skewed ({skew_val:.2f}) — consider log/sqrt transform.")
    if outlier_pct > 5:
        insights.append(f"⚠️  '{col}' has {outlier_pct:.1f}% outliers (IQR method).")
    if not is_normal:
        insights.append(f"ℹ️  '{col}' is NOT normally distributed (p={p_normal:.4f}).")

    # ── Plot: histogram + boxplot + target relationship ──────────────────────
    has_target = target in df.columns
    ncols = 3 if has_target else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    fig.suptitle(col, fontsize=13, fontweight="bold")

    # Histogram with KDE
    sns.histplot(series, kde=True, ax=axes[0], color="steelblue")
    axes[0].axvline(mean_val,   color="red",    linestyle="--", label=f"mean={mean_val:.2f}")
    axes[0].axvline(median_val, color="orange", linestyle="--", label=f"median={median_val:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distribution")

    # Boxplot
    sns.boxplot(y=series, ax=axes[1], color="lightblue")
    axes[1].set_title(f"Boxplot  |  outliers≈{outlier_pct:.1f}%")

    # Target relationship
    if has_target:
        if problem_type == "classification":
            sns.boxplot(x=df[target].astype(str), y=df[col], ax=axes[2], palette="Set2")
            axes[2].set_title(f"vs Target (class)")
        else:
            sns.scatterplot(x=df[col], y=df[target], ax=axes[2],
                            alpha=0.4, color="coral", s=15)
            # Regression line
            m, b, r, *_ = stats.linregress(df[col].fillna(mean_val), df[target])
            x_line = np.linspace(series.min(), series.max(), 100)
            axes[2].plot(x_line, m * x_line + b, color="red", linewidth=1.5,
                         label=f"r={r:.2f}")
            axes[2].legend(fontsize=8)
            axes[2].set_title(f"vs Target (r={r:.2f})")

    _save(fig, f"{col}_analysis.png", plots)

    return {
        "type": "numerical",
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outlier_pct": outlier_pct,
        "is_normal": is_normal,
        "normality_p": float(p_normal),
    }


def _analyse_categorical(df: pd.DataFrame, col: str, target: str,
                         problem_type: str, plots: list, insights: list) -> dict:
    series = df[col].dropna()
    nunique   = int(series.nunique())
    top_vals  = series.value_counts(normalize=True).head(10)
    top_dict  = {str(k): round(float(v), 4) for k, v in top_vals.items()}
    dominance = float(top_vals.iloc[0]) if len(top_vals) > 0 else 0.0

    if dominance > 0.90:
        insights.append(f"⚠️  '{col}' is near-constant — top value covers {dominance*100:.1f}% of rows.")
    if nunique > 50:
        insights.append(f"⚠️  '{col}' has high cardinality ({nunique} unique) — consider encoding strategy.")

    has_target = target in df.columns
    ncols = 2 if has_target else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(col, fontsize=13, fontweight="bold")

    # Value counts bar
    top_vals_abs = series.value_counts().head(10)
    sns.barplot(x=top_vals_abs.values, y=top_vals_abs.index.astype(str),
                ax=axes[0], palette="Blues_r", orient="h")
    axes[0].set_title(f"Top values  |  {nunique} unique")
    axes[0].set_xlabel("Count")

    # Target relationship
    if has_target:
        if problem_type == "classification":
            ct = pd.crosstab(df[col], df[target], normalize="index").head(10)
            ct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
            axes[1].set_title("Class distribution by category")
            axes[1].set_xlabel("")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            plot_data = df[[col, target]].copy()
            plot_data[col] = plot_data[col].astype(str)
            top_cats = top_vals_abs.head(8).index.astype(str).tolist()
            plot_data = plot_data[plot_data[col].isin(top_cats)]
            sns.boxplot(x=col, y=target, data=plot_data, ax=axes[1], palette="Set3")
            axes[1].set_title("Target distribution by category")
            axes[1].tick_params(axis="x", rotation=45)

    _save(fig, f"{col}_analysis.png", plots)

    return {
        "type": "categorical",
        "unique_values": nunique,
        "top_values": top_dict,
        "dominance": dominance,
    }


def _correlation_analysis(df: pd.DataFrame, target: str,
                          plots: list, insights: list) -> dict:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return {}

    corr = num_df.corr()

    # Full heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(num_df.columns)), max(6, len(num_df.columns) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap (lower triangle)", fontsize=12)
    _save(fig, "correlation_heatmap.png", plots)

    # Target correlation bar chart
    target_corr = {}
    if target in num_df.columns:
        target_corr = (
            corr[target]
            .drop(target, errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .to_dict()
        )
        fig2, ax2 = plt.subplots(figsize=(8, max(4, len(target_corr) * 0.4)))
        vals = pd.Series(target_corr)
        colors = ["tomato" if v > 0.5 else "steelblue" for v in vals]
        vals.plot(kind="barh", ax=ax2, color=colors)
        ax2.axvline(0.5, color="red", linestyle="--", linewidth=1, label="r=0.5")
        ax2.set_title(f"Feature Correlation with Target '{target}'")
        ax2.legend()
        _save(fig2, "target_correlation.png", plots)

        # Insight: highly correlated features
        strong = [f for f, v in target_corr.items() if v > 0.5]
        weak   = [f for f, v in target_corr.items() if v < 0.05]
        if strong:
            insights.append(f"✅ Strongly correlated with target: {', '.join(strong)}")
        if weak:
            insights.append(f"ℹ️  Weakly correlated with target (may be dropped): {', '.join(weak)}")

    # Multicollinearity warning
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = [
        (c1, c2, round(upper.loc[c1, c2], 2))
        for c1 in upper.columns
        for c2 in upper.columns
        if pd.notna(upper.loc[c1, c2]) and abs(upper.loc[c1, c2]) > 0.85
    ]
    for c1, c2, r in high_pairs:
        insights.append(f"⚠️  Multicollinearity: '{c1}' & '{c2}' are highly correlated (r={r}).")

    return {"target_correlations": target_corr, "high_collinear_pairs": high_pairs}


def _missing_value_analysis(df: pd.DataFrame, plots: list, insights: list) -> dict:
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        insights.append("✅ No missing values found in the dataset.")
        return {}

    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
    missing_pct = missing * 100
    colors = ["tomato" if v > 20 else "steelblue" for v in missing_pct]
    missing_pct.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(20, color="red", linestyle="--", linewidth=1, label="20% threshold")
    ax.set_title("Missing Value % per Column")
    ax.set_xlabel("Missing %")
    ax.legend()
    _save(fig, "missing_values.png", plots)

    high_missing = missing[missing > 0.4].index.tolist()
    if high_missing:
        insights.append(f"⚠️  Columns with >40% missing (consider dropping): {', '.join(high_missing)}")

    return {col: round(float(pct * 100), 2) for col, pct in missing.items()}


def _target_analysis(df: pd.DataFrame, target: str,
                     problem_type: str, plots: list, insights: list):
    if target not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"Target: '{target}'", fontsize=13, fontweight="bold")

    if problem_type == "classification":
        vc = df[target].value_counts()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="Set2")
        ax.set_title("Class Distribution")
        ax.set_ylabel("Count")

        # Imbalance check
        ratio = vc.min() / vc.max()
        if ratio < 0.2:
            insights.append(f"⚠️  Target is imbalanced! Minority/majority ratio = {ratio:.2f}. Consider oversampling.")
        else:
            insights.append(f"✅ Target class balance looks reasonable (ratio={ratio:.2f}).")
    else:
        sns.histplot(df[target].dropna(), kde=True, ax=ax, color="coral")
        ax.set_title("Target Distribution")
        skew = df[target].skew()
        if abs(skew) > 1:
            insights.append(f"⚠️  Target '{target}' is skewed ({skew:.2f}) — consider log transform.")

    _save(fig, f"target_{target}_distribution.png", plots)


# ── Main agent ────────────────────────────────────────────────────────────────

def eda_agent(state: dict) -> dict:
    if state.get("error"):
        return state

    df: pd.DataFrame | None = state.get("raw_data")
    if df is None:
        return {**state, "error": "EDA agent expected 'raw_data', but no dataset is loaded."}

    target: str      = state["target_column"]
    problem_type: str = state.get("problem_type", "classification")

    insights       = []
    plots          = []
    feature_summary = {}

    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Target distribution
    _target_analysis(df, target, problem_type, plots, insights)

    # 2. Missing value overview
    missing_summary = _missing_value_analysis(df, plots, insights)

    # 3. Per-feature analysis
    for col in df.columns:
        if col == target:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_summary[col] = _analyse_numerical(
                    df, col, target, problem_type, plots, insights)
            else:
                feature_summary[col] = _analyse_categorical(
                    df, col, target, problem_type, plots, insights)
        except Exception as e:
            insights.append(f"❌ Could not analyse '{col}': {e}")

    # 4. Correlation analysis
    corr_summary = _correlation_analysis(df, target, plots, insights)

    return {
        **state,
        "eda_report": {
            "insights":        insights,
            "plots":           plots,
            "feature_summary": feature_summary,
            "missing_summary": missing_summary,
            "correlation":     corr_summary,
        }
    }
