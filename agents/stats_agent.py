# agents/stats_agent.py

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import (
    chi2_contingency, f_oneway, kruskal,
    mannwhitneyu, pointbiserialr, spearmanr, pearsonr,
    levene, shapiro, ttest_ind,
)
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

ALPHA = 0.05  # significance threshold


# ─────────────────────────────────────────────
# Effect Size Helpers
# ─────────────────────────────────────────────

def _cramers_v(contingency: pd.DataFrame) -> float:
    """Effect size for chi-square (categorical vs categorical)."""
    chi2 = chi2_contingency(contingency)[0]
    n    = contingency.values.sum()
    r, k = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else 0.0


def _eta_squared(groups: list) -> float:
    """Effect size for ANOVA (numeric vs categorical target)."""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum(((v - grand_mean) ** 2) for g in groups for v in g)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """Effect size for two-group comparison."""
    pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    return float((g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _epsilon_squared(stat: float, n: int, k: int) -> float:
    """Effect size for Kruskal-Wallis."""
    return float((stat - k + 1) / (n - k)) if n > k else 0.0


def _interpret_effect(size: float, metric: str) -> str:
    """Human-readable effect size label."""
    if metric in ("cramers_v", "eta_squared", "epsilon_squared"):
        if size < 0.10: return "negligible"
        if size < 0.30: return "small"
        if size < 0.50: return "medium"
        return "large"
    if metric == "cohens_d":
        abs_d = abs(size)
        if abs_d < 0.20: return "negligible"
        if abs_d < 0.50: return "small"
        if abs_d < 0.80: return "medium"
        return "large"
    if metric in ("pearson_r", "spearman_r", "point_biserial_r"):
        abs_r = abs(size)
        if abs_r < 0.10: return "negligible"
        if abs_r < 0.30: return "small"
        if abs_r < 0.50: return "medium"
        return "large"
    return "unknown"


# ─────────────────────────────────────────────
# Pre-test Checks
# ─────────────────────────────────────────────

def _check_normality(series: pd.Series) -> bool:
    """Shapiro-Wilk on a sample; returns True if normal."""
    clean = series.dropna()
    if len(clean) < 8:
        return False
    sample = clean.sample(min(500, len(clean)), random_state=42)
    _, p   = shapiro(sample)
    return bool(p > 0.05)


def _check_variance_homogeneity(groups: list) -> bool:
    """Levene's test for equal variances; returns True if homogeneous."""
    if len(groups) < 2:
        return True
    try:
        _, p = levene(*groups)
        return bool(p > 0.05)
    except Exception:
        return False


# ─────────────────────────────────────────────
# Individual Test Functions
# ─────────────────────────────────────────────

def _test_numeric_vs_classification(
    series: pd.Series, target: pd.Series
) -> dict:
    """
    For numeric feature vs. categorical target.
    Selects the best test automatically:
      - 2 groups + normal + equal var  → Student's t-test
      - 2 groups + non-normal          → Mann-Whitney U
      - k groups + normal + equal var  → One-way ANOVA
      - k groups + otherwise           → Kruskal-Wallis
    """
    target_vals = target.dropna().unique()
    groups = [series[target == v].dropna() for v in target_vals]
    groups = [g for g in groups if len(g) >= 3]

    if len(groups) < 2:
        return {"skipped": "insufficient_groups"}

    n_groups  = len(groups)
    n_total   = sum(len(g) for g in groups)
    is_normal = all(_check_normality(g) for g in groups)
    equal_var = _check_variance_homogeneity(groups)

    result = {
        "n_groups":    n_groups,
        "group_sizes": {str(v): len(g) for v, g in zip(target_vals, groups)},
        "group_means": {str(v): round(float(g.mean()), 4) for v, g in zip(target_vals, groups)},
        "group_stds":  {str(v): round(float(g.std()), 4)  for v, g in zip(target_vals, groups)},
        "normality_assumption": is_normal,
        "equal_variance_assumption": equal_var,
    }

    if n_groups == 2:
        g1, g2 = groups[0], groups[1]
        if is_normal and equal_var:
            stat, p = ttest_ind(g1, g2, equal_var=True)
            result.update({
                "test":        "students_t",
                "statistic":   round(float(stat), 4),
                "p_value":     round(float(p), 4),
                "effect_size": round(_cohens_d(g1, g2), 4),
                "effect_metric": "cohens_d",
            })
        elif is_normal and not equal_var:
            stat, p = ttest_ind(g1, g2, equal_var=False)
            result.update({
                "test":        "welch_t",
                "statistic":   round(float(stat), 4),
                "p_value":     round(float(p), 4),
                "effect_size": round(_cohens_d(g1, g2), 4),
                "effect_metric": "cohens_d",
            })
        else:
            stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            result.update({
                "test":        "mann_whitney_u",
                "statistic":   round(float(stat), 4),
                "p_value":     round(float(p), 4),
                "effect_size": round(_cohens_d(g1, g2), 4),
                "effect_metric": "cohens_d",
            })
    else:
        if is_normal and equal_var:
            stat, p = f_oneway(*groups)
            result.update({
                "test":        "anova",
                "statistic":   round(float(stat), 4),
                "p_value":     round(float(p), 4),
                "effect_size": round(_eta_squared(groups), 4),
                "effect_metric": "eta_squared",
            })
        else:
            stat, p = kruskal(*groups)
            result.update({
                "test":        "kruskal_wallis",
                "statistic":   round(float(stat), 4),
                "p_value":     round(float(p), 4),
                "effect_size": round(_epsilon_squared(stat, n_total, n_groups), 4),
                "effect_metric": "epsilon_squared",
            })

    result["effect_label"] = _interpret_effect(
        result.get("effect_size", 0), result.get("effect_metric", "")
    )
    return result


def _test_numeric_vs_regression(
    series: pd.Series, target: pd.Series
) -> dict:
    """
    For numeric feature vs. numeric target.
    Runs both Pearson and Spearman; chooses the appropriate one
    based on normality of both series.
    """
    df_pair = pd.DataFrame({"x": series, "y": target}).dropna()
    if len(df_pair) < 10:
        return {"skipped": "insufficient_data"}

    x, y         = df_pair["x"], df_pair["y"]
    x_normal     = _check_normality(x)
    y_normal     = _check_normality(y)
    both_normal  = x_normal and y_normal

    pearson_r, pearson_p   = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    # Primary test chosen by normality
    primary = "pearson" if both_normal else "spearman"
    p_val   = pearson_p if both_normal else spearman_p
    r_val   = pearson_r if both_normal else spearman_r

    return {
        "test":           primary,
        "p_value":        round(float(p_val), 4),
        "statistic":      round(float(r_val), 4),
        "effect_size":    round(float(abs(r_val)), 4),
        "effect_metric":  f"{primary}_r",
        "effect_label":   _interpret_effect(abs(r_val), f"{primary}_r"),
        "pearson_r":      round(float(pearson_r), 4),
        "pearson_p":      round(float(pearson_p), 4),
        "spearman_r":     round(float(spearman_r), 4),
        "spearman_p":     round(float(spearman_p), 4),
        "x_normal":       x_normal,
        "y_normal":       y_normal,
    }


def _test_categorical_vs_classification(
    series: pd.Series, target: pd.Series
) -> dict:
    """Chi-square test with Cramér's V effect size."""
    contingency = pd.crosstab(series, target)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"skipped": "insufficient_categories"}

    # Check expected frequency assumption (80% cells ≥ 5)
    chi2, p, dof, expected = chi2_contingency(contingency)
    pct_low_expected = float((expected < 5).mean())
    assumption_ok    = pct_low_expected < 0.20

    v = _cramers_v(contingency)

    return {
        "test":             "chi_square",
        "statistic":        round(float(chi2), 4),
        "p_value":          round(float(p), 4),
        "degrees_of_freedom": int(dof),
        "effect_size":      round(v, 4),
        "effect_metric":    "cramers_v",
        "effect_label":     _interpret_effect(v, "cramers_v"),
        "expected_freq_assumption_ok": assumption_ok,
        "pct_cells_low_expected": round(pct_low_expected * 100, 1),
        "contingency_shape": list(contingency.shape),
    }


def _test_categorical_vs_regression(
    series: pd.Series, target: pd.Series
) -> dict:
    """
    For categorical feature vs. numeric target.
    ANOVA or Kruskal-Wallis depending on normality/variance checks.
    """
    groups_map = {
        cat: target[series == cat].dropna()
        for cat in series.dropna().unique()
    }
    groups_map = {k: v for k, v in groups_map.items() if len(v) >= 3}

    if len(groups_map) < 2:
        return {"skipped": "insufficient_groups"}

    groups    = list(groups_map.values())
    n_total   = sum(len(g) for g in groups)
    is_normal = all(_check_normality(g) for g in groups)
    equal_var = _check_variance_homogeneity(groups)

    result = {
        "n_groups":    len(groups),
        "group_means": {str(k): round(float(v.mean()), 4) for k, v in groups_map.items()},
        "normality_assumption":     is_normal,
        "equal_variance_assumption": equal_var,
    }

    if is_normal and equal_var:
        stat, p = f_oneway(*groups)
        result.update({
            "test":        "anova",
            "statistic":   round(float(stat), 4),
            "p_value":     round(float(p), 4),
            "effect_size": round(_eta_squared(groups), 4),
            "effect_metric": "eta_squared",
        })
    else:
        stat, p = kruskal(*groups)
        result.update({
            "test":        "kruskal_wallis",
            "statistic":   round(float(stat), 4),
            "p_value":     round(float(p), 4),
            "effect_size": round(_epsilon_squared(stat, n_total, len(groups)), 4),
            "effect_metric": "epsilon_squared",
        })

    result["effect_label"] = _interpret_effect(
        result.get("effect_size", 0), result.get("effect_metric", "")
    )
    return result


def _test_binary_vs_target(
    series: pd.Series, target: pd.Series, problem_type: str
) -> dict:
    """Point-biserial correlation for binary feature vs. numeric target."""
    if problem_type != "regression":
        return {}
    df_pair = pd.DataFrame({"x": series, "y": target}).dropna()
    if df_pair["x"].nunique() != 2:
        return {}
    encoded = (df_pair["x"] == df_pair["x"].unique()[0]).astype(int)
    r, p    = pointbiserialr(encoded, df_pair["y"])
    return {
        "test":          "point_biserial",
        "statistic":     round(float(r), 4),
        "p_value":       round(float(p), 4),
        "effect_size":   round(float(abs(r)), 4),
        "effect_metric": "point_biserial_r",
        "effect_label":  _interpret_effect(abs(r), "point_biserial_r"),
    }


# ─────────────────────────────────────────────
# Multicollinearity Between Features
# ─────────────────────────────────────────────

def _feature_intercorrelation(df: pd.DataFrame, cols: list) -> list:
    """
    Spearman correlation between all numeric feature pairs.
    Flags pairs with |r| > 0.85.
    """
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    flags    = []
    for c1, c2 in combinations(num_cols, 2):
        pair = df[[c1, c2]].dropna()
        if len(pair) < 10:
            continue
        r, p = spearmanr(pair[c1], pair[c2])
        if abs(r) > 0.85:
            flags.append({
                "col_a":      c1,
                "col_b":      c2,
                "spearman_r": round(float(r), 4),
                "p_value":    round(float(p), 4),
                "flag":       "high_multicollinearity",
            })
    return flags


# ─────────────────────────────────────────────
# FDR Correction (Benjamini-Hochberg)
# ─────────────────────────────────────────────

def _bh_correction(test_details: dict) -> dict:
    """
    Apply Benjamini-Hochberg FDR correction to all p-values.
    Adds 'p_adjusted' and 'significant_after_correction' to each entry.
    """
    cols   = [c for c, d in test_details.items() if "p_value" in d]
    pvals  = np.array([test_details[c]["p_value"] for c in cols])
    n      = len(pvals)

    if n == 0:
        return test_details

    sorted_idx  = np.argsort(pvals)
    sorted_p    = pvals[sorted_idx]
    bh_critical = (np.arange(1, n + 1) / n) * ALPHA

    # Find largest k where p(k) <= (k/n) * alpha
    significant = sorted_p <= bh_critical
    if significant.any():
        max_k = np.where(significant)[0].max()
        reject = np.zeros(n, dtype=bool)
        reject[sorted_idx[:max_k + 1]] = True
    else:
        reject = np.zeros(n, dtype=bool)

    # Adjusted p-values (Benjamini-Hochberg)
    adj_p    = np.minimum(1, sorted_p * n / np.arange(1, n + 1))
    adj_p    = np.minimum.accumulate(adj_p[::-1])[::-1]
    adj_p_ordered = np.empty(n)
    adj_p_ordered[sorted_idx] = adj_p

    for i, col in enumerate(cols):
        test_details[col]["p_adjusted"] = round(float(adj_p_ordered[i]), 4)
        test_details[col]["significant_after_fdr"] = bool(reject[i])

    return test_details


# ─────────────────────────────────────────────
# Feature Ranking
# ─────────────────────────────────────────────

def _rank_features(test_details: dict) -> list:
    """
    Rank features by: (1) significance after FDR, (2) effect size descending.
    Returns ordered list of dicts.
    """
    rows = []
    for col, d in test_details.items():
        if "skipped" in d:
            continue
        rows.append({
            "feature":      col,
            "test":         d.get("test", "—"),
            "p_value":      d.get("p_value", 1.0),
            "p_adjusted":   d.get("p_adjusted", 1.0),
            "effect_size":  d.get("effect_size", 0.0),
            "effect_label": d.get("effect_label", "—"),
            "significant":  d.get("p_value", 1.0) < ALPHA,
            "significant_after_fdr": d.get("significant_after_fdr", False),
        })

    rows.sort(key=lambda x: (-int(x["significant_after_fdr"]), -x["effect_size"]))
    return rows


# ─────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────

def stats_agent(state: dict) -> dict:
    if state.get("error"):
        return state

    df = state.get("raw_data")
    if df is None:
        return {**state, "error": "Stats agent expected 'raw_data', but no dataset is loaded."}

    target       = state["target_column"]
    problem_type = state.get("problem_type", "classification")

    if target not in df.columns:
        return {**state, "error": f"Target '{target}' not in DataFrame."}

    target_series = df[target].dropna()
    feature_cols  = [c for c in df.columns if c != target]

    test_details   = {}
    skipped        = []

    for col in feature_cols:
        series = df[col]

        # Skip constant or all-missing columns
        if series.nunique() <= 1 or series.isnull().all():
            skipped.append({"col": col, "reason": "constant_or_all_missing"})
            continue

        try:
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_cat     = series.dtype == object or str(series.dtype) == "category"
            n_unique   = series.nunique()

            # Binary numeric (treat as categorical for classification)
            if is_numeric and n_unique == 2 and problem_type == "regression":
                result = _test_binary_vs_target(series, target_series, problem_type)
                if result:
                    test_details[col] = result
                    continue

            if is_numeric:
                if problem_type == "classification":
                    result = _test_numeric_vs_classification(series, target_series)
                else:
                    result = _test_numeric_vs_regression(series, target_series)

            elif is_cat:
                if problem_type == "classification":
                    result = _test_categorical_vs_classification(series, target_series)
                else:
                    result = _test_categorical_vs_regression(series, target_series)
            else:
                skipped.append({"col": col, "reason": "unsupported_dtype"})
                continue

            if "skipped" in result:
                skipped.append({"col": col, "reason": result["skipped"]})
            else:
                test_details[col] = result

        except Exception as e:
            skipped.append({"col": col, "reason": f"error: {e}"})

    # ── FDR Correction ────────────────────────
    test_details = _bh_correction(test_details)

    # ── Feature ranking ───────────────────────
    ranked_features = _rank_features(test_details)

    # ── Significance buckets ──────────────────
    significant         = [r["feature"] for r in ranked_features if r["significant"]]
    significant_fdr     = [r["feature"] for r in ranked_features if r["significant_after_fdr"]]
    insignificant       = [r["feature"] for r in ranked_features if not r["significant"]]

    # ── Effect size buckets ───────────────────
    large_effect  = [r["feature"] for r in ranked_features if r["effect_label"] == "large"]
    medium_effect = [r["feature"] for r in ranked_features if r["effect_label"] == "medium"]
    small_effect  = [r["feature"] for r in ranked_features if r["effect_label"] in ("small", "negligible")]

    # ── Multicollinearity ─────────────────────
    intercorr_flags = _feature_intercorrelation(df, feature_cols)

    # ── Warnings ─────────────────────────────
    warnings_list = []
    if len(significant) == 0:
        warnings_list.append("⚠️ No features are statistically significant — check data quality or target encoding.")
    if len(significant) - len(significant_fdr) > 3:
        warnings_list.append(
            f"⚠️ {len(significant) - len(significant_fdr)} features lose significance after FDR correction — potential false positives."
        )
    for col, d in test_details.items():
        if d.get("expected_freq_assumption_ok") is False:
            warnings_list.append(
                f"⚠️ '{col}': chi-square may be unreliable ({d.get('pct_cells_low_expected', '?')}% cells with expected < 5)."
            )
        if d.get("effect_label") == "negligible" and d.get("p_value", 1) < ALPHA:
            warnings_list.append(
                f"ℹ️ '{col}' is statistically significant but has negligible effect size — likely a large-sample artefact."
            )
    for flag in intercorr_flags:
        warnings_list.append(
            f"⚠️ Multicollinearity: '{flag['col_a']}' & '{flag['col_b']}' (Spearman r={flag['spearman_r']})."
        )

    return {
        **state,
        "stats_report": {
            # Core results
            "significant":          significant,
            "significant_after_fdr": significant_fdr,
            "insignificant":        insignificant,
            "skipped":              skipped,

            # Ranked feature table
            "ranked_features":      ranked_features,

            # Effect size buckets
            "large_effect_features":  large_effect,
            "medium_effect_features": medium_effect,
            "small_effect_features":  small_effect,

            # Per-feature test details
            "details":              test_details,

            # Multicollinearity
            "multicollinearity_flags": intercorr_flags,

            # Meta
            "alpha":                ALPHA,
            "n_features_tested":    len(test_details),
            "n_features_skipped":   len(skipped),
            "problem_type":         problem_type,
            "warnings":             warnings_list,
        }
    }
