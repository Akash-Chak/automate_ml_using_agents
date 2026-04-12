# agents/profiling_agent.py

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _infer_semantic_type(series: pd.Series, col_name: str) -> str:
    """
    Goes beyond dtype to infer real-world semantic type:
    id, datetime, boolean, binary, ordinal, categorical, continuous, text, constant.
    """
    col_lower = col_name.lower()
    n         = len(series.dropna())
    nunique   = series.nunique()

    if nunique <= 1:
        return "constant"

    # Boolean / binary
    if set(series.dropna().unique()).issubset({0, 1, True, False, "yes", "no", "y", "n", "true", "false"}):
        return "boolean"

    # ID-like column
    if any(kw in col_lower for kw in ["_id", "id_", " id", "uuid", "key"]):
        if nunique / max(n, 1) > 0.9:
            return "id"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if not pd.api.types.is_numeric_dtype(series):
        try:
            pd.to_datetime(series.dropna().head(50), infer_datetime_format=True, errors="raise")
            return "datetime_string"
        except Exception:
            pass

    # Text (long strings)
    if series.dtype == object:
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 50:
            return "text"

    # Numeric types
    if pd.api.types.is_numeric_dtype(series):
        if nunique == 2:
            return "binary_numeric"
        ratio = nunique / max(n, 1)
        if nunique <= 15 or ratio < 0.05:
            return "ordinal_or_categorical_numeric"
        return "continuous"

    # Categorical
    ratio = nunique / max(n, 1)
    if ratio < 0.5:
        return "categorical"

    return "text"


def _outlier_analysis(series: pd.Series) -> dict:
    """IQR + Z-score dual outlier detection."""
    clean = series.dropna()
    if len(clean) < 4:
        return {}

    # IQR method
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr    = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask     = (clean < lower) | (clean > upper)
    iqr_count    = int(iqr_mask.sum())

    # Z-score method
    z_scores  = np.abs(scipy_stats.zscore(clean))
    z_count   = int((z_scores > 3).sum())

    return {
        "iqr_outlier_count": iqr_count,
        "iqr_outlier_pct":   round(iqr_count / len(clean) * 100, 2),
        "zscore_outlier_count": z_count,
        "zscore_outlier_pct":   round(z_count / len(clean) * 100, 2),
        "iqr_lower_bound":  round(float(lower), 4),
        "iqr_upper_bound":  round(float(upper), 4),
    }


def _normality_test(series: pd.Series) -> dict:
    """Shapiro-Wilk (small) or D'Agostino-Pearson (large)."""
    clean = series.dropna()
    if len(clean) < 8:
        return {"normality_test": "insufficient_data"}

    if len(clean) <= 2000:
        sample = clean.sample(min(len(clean), 500), random_state=42)
        stat, p = scipy_stats.shapiro(sample)
        test_name = "shapiro_wilk"
    else:
        stat, p = scipy_stats.normaltest(clean)
        test_name = "dagostino_pearson"

    return {
        "normality_test":    test_name,
        "normality_stat":    round(float(stat), 4),
        "normality_p":       round(float(p), 4),
        "is_normal":         bool(p > 0.05),
    }


def _profile_numeric(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {"type": "numeric", "note": "all_missing"}

    q1, med, q3 = clean.quantile([0.25, 0.50, 0.75])
    result = {
        "type":        "numeric",
        "count":       int(len(clean)),
        "mean":        round(float(clean.mean()), 4),
        "median":      round(float(med), 4),
        "std":         round(float(clean.std()), 4),
        "variance":    round(float(clean.var()), 4),
        "min":         round(float(clean.min()), 4),
        "max":         round(float(clean.max()), 4),
        "range":       round(float(clean.max() - clean.min()), 4),
        "q1":          round(float(q1), 4),
        "q3":          round(float(q3), 4),
        "iqr":         round(float(q3 - q1), 4),
        "skewness":    round(float(clean.skew()), 4),
        "kurtosis":    round(float(clean.kurt()), 4),
        "cv":          round(float(clean.std() / clean.mean()), 4) if clean.mean() != 0 else None,
        "zeros_count": int((clean == 0).sum()),
        "zeros_pct":   round(float((clean == 0).mean() * 100), 2),
        "negatives_count": int((clean < 0).sum()),
    }

    result.update(_outlier_analysis(clean))
    result.update(_normality_test(clean))

    # Distribution shape label
    skew = result["skewness"]
    kurt = result["kurtosis"]
    if abs(skew) < 0.5 and abs(kurt) < 1:
        result["distribution_shape"] = "normal-like"
    elif skew > 1:
        result["distribution_shape"] = "right-skewed"
    elif skew < -1:
        result["distribution_shape"] = "left-skewed"
    elif kurt > 3:
        result["distribution_shape"] = "leptokurtic (heavy tails)"
    elif kurt < -1:
        result["distribution_shape"] = "platykurtic (light tails)"
    else:
        result["distribution_shape"] = "moderate"

    return result


def _profile_categorical(series: pd.Series) -> dict:
    clean  = series.dropna()
    vc     = clean.value_counts()
    n      = len(clean)

    top5   = vc.head(5).to_dict()
    top5   = {str(k): int(v) for k, v in top5.items()}

    # Shannon entropy (diversity measure)
    probs   = vc / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))

    return {
        "type":           "categorical",
        "count":          n,
        "unique_values":  int(series.nunique()),
        "top_5_values":   top5,
        "top_value":      str(vc.index[0]) if len(vc) > 0 else None,
        "top_value_pct":  round(float(vc.iloc[0] / n * 100), 2) if len(vc) > 0 else None,
        "bottom_value":   str(vc.index[-1]) if len(vc) > 0 else None,
        "bottom_value_count": int(vc.iloc[-1]) if len(vc) > 0 else None,
        "entropy":        round(entropy, 4),
        "dominance":      round(float(vc.iloc[0] / n), 4) if len(vc) > 0 else None,
        "cardinality_ratio": round(float(series.nunique() / max(n, 1)), 4),
    }


def _profile_datetime(series: pd.Series) -> dict:
    try:
        dt = pd.to_datetime(series.dropna(), infer_datetime_format=True)
        return {
            "type":          "datetime",
            "count":         int(len(dt)),
            "min":           str(dt.min()),
            "max":           str(dt.max()),
            "range_days":    int((dt.max() - dt.min()).days),
            "unique_dates":  int(dt.nunique()),
        }
    except Exception:
        return {"type": "datetime", "note": "parse_error"}


# ─────────────────────────────────────────────
# Dataset-level analyses
# ─────────────────────────────────────────────

def _missing_analysis(df: pd.DataFrame) -> dict:
    missing      = df.isnull().sum()
    missing_pct  = df.isnull().mean() * 100
    total_cells  = df.shape[0] * df.shape[1]
    total_missing = int(missing.sum())

    per_column = {
        col: {
            "missing_count": int(missing[col]),
            "missing_pct":   round(float(missing_pct[col]), 2),
        }
        for col in df.columns if missing[col] > 0
    }

    # Missing patterns: columns that are always missing together
    missing_pairs = []
    missing_cols  = [c for c in df.columns if missing[c] > 0]
    for i in range(len(missing_cols)):
        for j in range(i + 1, len(missing_cols)):
            c1, c2 = missing_cols[i], missing_cols[j]
            both   = (df[c1].isnull() & df[c2].isnull()).sum()
            if both > 0:
                missing_pairs.append({
                    "col_a": c1, "col_b": c2, "co_missing_count": int(both)
                })

    return {
        "total_missing_cells":   total_missing,
        "total_missing_pct":     round(total_missing / max(total_cells, 1) * 100, 2),
        "columns_with_missing":  int((missing > 0).sum()),
        "columns_all_missing":   int((missing == df.shape[0]).sum()),
        "per_column":            per_column,
        "co_missing_pairs":      missing_pairs[:10],  # top 10
    }


def _duplicate_analysis(df: pd.DataFrame) -> dict:
    n_dup     = int(df.duplicated().sum())
    n_dup_col = {col: int(df[col].duplicated().sum()) for col in df.columns}
    return {
        "duplicate_rows":        n_dup,
        "duplicate_rows_pct":    round(n_dup / max(len(df), 1) * 100, 2),
        "columns_with_duplicate_values": {
            k: v for k, v in n_dup_col.items() if v > 0
        },
    }


def _correlation_flags(df: pd.DataFrame) -> list:
    """Flag highly correlated numeric feature pairs."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return []

    corr   = num_df.corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    flags  = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > 0.85:
                flags.append({
                    "col_a": row, "col_b": col,
                    "pearson_r": round(float(corr.loc[row, col]), 4),
                    "flag": "high_multicollinearity"
                })
    return flags


def _data_quality_score(df: pd.DataFrame, column_summary: dict) -> dict:
    """
    Simple composite data quality score (0–100) across four dimensions.
    """
    n_rows, n_cols = df.shape

    # Completeness: % non-missing cells
    completeness = float((1 - df.isnull().mean().mean()) * 100)

    # Uniqueness: penalise duplicate rows
    uniqueness = float((1 - df.duplicated().mean()) * 100)

    # Consistency: penalise constant/id columns
    sem_types   = [v.get("semantic_type", "") for v in column_summary.values()]
    bad_cols    = sum(1 for t in sem_types if t in ("constant", "id"))
    consistency = float(max(0, (1 - bad_cols / max(n_cols, 1)) * 100))

    # Validity: penalise columns with >20% outliers
    outlier_heavy = sum(
        1 for v in column_summary.values()
        if v.get("iqr_outlier_pct", 0) > 20
    )
    validity = float(max(0, (1 - outlier_heavy / max(n_cols, 1)) * 100))

    overall = round((completeness + uniqueness + consistency + validity) / 4, 1)

    return {
        "overall":      overall,
        "completeness": round(completeness, 1),
        "uniqueness":   round(uniqueness, 1),
        "consistency":  round(consistency, 1),
        "validity":     round(validity, 1),
        "grade":        "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D",
    }


def _generate_warnings(df: pd.DataFrame, column_summary: dict,
                        missing: dict, duplicates: dict) -> list:
    """Human-readable actionable warnings for downstream agents."""
    warnings_list = []

    # Dataset-level
    if duplicates["duplicate_rows_pct"] > 1:
        warnings_list.append(
            f"⚠️ {duplicates['duplicate_rows']} duplicate rows ({duplicates['duplicate_rows_pct']}%) — consider deduplication."
        )
    if missing["total_missing_pct"] > 20:
        warnings_list.append(
            f"⚠️ Dataset has {missing['total_missing_pct']}% missing values overall — imputation strategy needed."
        )

    # Column-level
    for col, info in column_summary.items():
        sem = info.get("semantic_type", "")
        miss_pct = info.get("missing_pct", 0)

        if sem == "constant":
            warnings_list.append(f"⚠️ '{col}' is constant — drop it.")
        if sem == "id":
            warnings_list.append(f"ℹ️ '{col}' looks like an ID column — exclude from modelling.")
        if miss_pct > 40:
            warnings_list.append(f"⚠️ '{col}' has {miss_pct:.1f}% missing — consider dropping.")
        if info.get("iqr_outlier_pct", 0) > 15:
            warnings_list.append(
                f"⚠️ '{col}' has {info['iqr_outlier_pct']}% outliers (IQR) — review or cap/floor."
            )
        if info.get("dominance", 0) > 0.95:
            warnings_list.append(
                f"⚠️ '{col}' is near-constant ({info['dominance']*100:.1f}% one value) — low signal."
            )
        if sem == "right-skewed" or info.get("distribution_shape") == "right-skewed":
            warnings_list.append(
                f"ℹ️ '{col}' is right-skewed — consider log/sqrt transform."
            )
        if info.get("zeros_pct", 0) > 30:
            warnings_list.append(
                f"ℹ️ '{col}' has {info['zeros_pct']}% zeros — check if structural or missing."
            )
        if info.get("cardinality_ratio", 0) > 0.9 and sem == "categorical":
            warnings_list.append(
                f"⚠️ '{col}' has very high cardinality ({info['unique_values']} unique) — consider hashing or embedding."
            )

    return warnings_list


# ─────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────

def profiling_agent(state: dict) -> dict:
    # ── Load data ────────────────────────────
    dataset_path = state.get("dataset_path")
    target       = state.get("target_column")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        return {**state, "error": f"Dataset not found: {dataset_path}"}
    except Exception as e:
        return {**state, "error": f"Failed to load dataset: {e}"}

    # Try parsing datetime columns automatically
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                df[col] = parsed
            except Exception:
                pass

    # ── Validate target ───────────────────────
    if target and target not in df.columns:
        return {**state, "error": f"Target column '{target}' not found in dataset. Available: {list(df.columns)}"}

    n_rows, n_cols = df.shape

    # ── Per-column profiling ──────────────────
    column_summary = {}

    for col in df.columns:
        series    = df[col]
        sem_type  = _infer_semantic_type(series, col)
        miss_count = int(series.isnull().sum())
        miss_pct   = round(float(series.isnull().mean() * 100), 2)

        col_info = {
            "dtype":         str(series.dtype),
            "semantic_type": sem_type,
            "missing_count": miss_count,
            "missing_pct":   miss_pct,
            "cardinality":   int(series.nunique()),
            "is_target":     col == target,
        }

        try:
            if pd.api.types.is_numeric_dtype(series) and sem_type not in ("boolean", "id", "constant"):
                col_info.update(_profile_numeric(series))
            elif sem_type in ("datetime", "datetime_string"):
                col_info.update(_profile_datetime(series))
            else:
                col_info.update(_profile_categorical(series))
        except Exception as e:
            col_info["profile_error"] = str(e)

        column_summary[col] = col_info

    # ── Dataset-level analyses ────────────────
    missing_info    = _missing_analysis(df)
    duplicate_info  = _duplicate_analysis(df)
    corr_flags      = _correlation_flags(df)
    quality_score   = _data_quality_score(df, column_summary)
    warnings_list   = _generate_warnings(df, column_summary, missing_info, duplicate_info)

    # ── Column type summary ───────────────────
    sem_counts = {}
    for info in column_summary.values():
        t = info.get("semantic_type", "unknown")
        sem_counts[t] = sem_counts.get(t, 0) + 1

    # ── Recommendations for downstream agents ─
    recommendations = {
        "drop_candidates":   [c for c, i in column_summary.items()
                               if i.get("semantic_type") in ("constant", "id")],
        "high_missing":      [c for c, i in column_summary.items()
                               if i.get("missing_pct", 0) > 40 and c != target],
        "skewed_columns":    [c for c, i in column_summary.items()
                               if abs(i.get("skewness", 0)) > 1],
        "high_cardinality":  [c for c, i in column_summary.items()
                               if i.get("semantic_type") == "categorical"
                               and i.get("cardinality_ratio", 0) > 0.5],
        "needs_encoding":    [c for c, i in column_summary.items()
                               if i.get("semantic_type") in ("categorical", "boolean")
                               and c != target],
        "outlier_columns":   [c for c, i in column_summary.items()
                               if i.get("iqr_outlier_pct", 0) > 10],
        "correlated_pairs":  corr_flags,
    }

    profiling_report = {
        # Dataset overview
        "num_rows":          n_rows,
        "num_columns":       n_cols,
        "memory_usage_mb":   round(float(df.memory_usage(deep=True).sum() / 1024 ** 2), 3),
        "total_cells":       n_rows * n_cols,

        # Column type breakdown
        "semantic_type_counts": sem_counts,
        "numeric_columns":   [c for c, i in column_summary.items() if i.get("type") == "numeric"],
        "categorical_columns": [c for c, i in column_summary.items() if i.get("type") == "categorical"],
        "datetime_columns":  [c for c, i in column_summary.items() if i.get("type") == "datetime"],

        # Core reports
        "column_summary":    column_summary,
        "missing":           missing_info,
        "duplicates":        duplicate_info,
        "data_quality_score": quality_score,

        # Actionable outputs
        "warnings":          warnings_list,
        "recommendations":   recommendations,
    }

    return {
        **state,"raw_data":df,"profiling_report": profiling_report,
    }