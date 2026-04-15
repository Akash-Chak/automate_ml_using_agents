import numpy as np
import pandas as pd
from scipy.stats import yeojohnson as scipy_yeojohnson
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "missing"


def _rank_lookup(stats_report: dict) -> dict:
    lookup = {}
    for row in stats_report.get("ranked_features", []):
        lookup[row["feature"]] = row
    return lookup


def _target_strength_lookup(eda_report: dict) -> dict:
    return eda_report.get("correlation", {}).get("target_correlations", {}) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_yeo_johnson(series: pd.Series) -> pd.Series:
    """Yeo-Johnson power transform — handles negatives."""
    clean = series.dropna().values
    if len(clean) == 0:
        return series
    transformed, _ = scipy_yeojohnson(clean)
    result = series.copy()
    result[~series.isna()] = transformed
    return result


def _apply_cyclical_encoding(df: pd.DataFrame, col: str,
                              fe_steps: list) -> tuple[pd.DataFrame, list]:
    """
    For a datetime column: extract year/month/day/dayofweek and
    add sin/cos cyclical encodings for month and day-of-week.
    Returns (df, new_col_names).
    """
    series = pd.to_datetime(df[col], errors="coerce")
    if series.notna().sum() == 0:
        return df, []

    new_cols = []

    # Numeric extractions
    df[f"{col}_year"]      = series.dt.year;      new_cols.append(f"{col}_year")
    df[f"{col}_month"]     = series.dt.month;     new_cols.append(f"{col}_month")
    df[f"{col}_day"]       = series.dt.day;       new_cols.append(f"{col}_day")
    df[f"{col}_dayofweek"] = series.dt.dayofweek; new_cols.append(f"{col}_dayofweek")

    # Cyclical: month (period=12)
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * series.dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * series.dt.month / 12)
    new_cols += [f"{col}_month_sin", f"{col}_month_cos"]

    # Cyclical: day-of-week (period=7)
    df[f"{col}_dow_sin"] = np.sin(2 * np.pi * series.dt.dayofweek / 7)
    df[f"{col}_dow_cos"] = np.cos(2 * np.pi * series.dt.dayofweek / 7)
    new_cols += [f"{col}_dow_sin", f"{col}_dow_cos"]

    df = df.drop(columns=[col], errors="ignore")
    fe_steps.append({
        "col": col, "action": "decompose", "method": "cyclical",
        "output_cols": new_cols,
        "reason": "datetime column decomposed with cyclical sin/cos encoding",
    })
    return df, new_cols


def _apply_target_encoding(X: pd.DataFrame, y: pd.Series,
                            col: str, n_folds: int = 5) -> pd.Series:
    """
    5-fold leave-one-out target mean encoding.
    Encodes in-fold to avoid target leakage.
    """
    global_mean = float(y.mean())
    encoded = pd.Series(global_mean, index=X.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        train_vals = X[col].iloc[train_idx]
        train_y    = y.iloc[train_idx]
        means      = train_y.groupby(train_vals.values).mean()

        val_vals   = X[col].iloc[val_idx]
        encoded.iloc[val_idx] = val_vals.map(means).fillna(global_mean).values

    return encoded


def _apply_interaction(X: pd.DataFrame, col_a: str, col_b: str,
                        itype: str, fe_steps: list) -> pd.DataFrame:
    """
    Create an interaction feature between two numeric columns.
    type: ratio | product | difference
    """
    if col_a not in X.columns or col_b not in X.columns:
        return X

    if itype == "ratio":
        name = f"{col_a}_div_{col_b}"
        denom = X[col_b].replace(0, np.nan)
        X[name] = X[col_a] / denom
        X[name] = X[name].fillna(0.0)
    elif itype == "product":
        name = f"{col_a}_x_{col_b}"
        X[name] = X[col_a] * X[col_b]
    elif itype == "difference":
        name = f"{col_a}_minus_{col_b}"
        X[name] = X[col_a] - X[col_b]
    else:
        return X

    fe_steps.append({
        "col": name, "action": "interaction",
        "method": itype,
        "col_a": col_a, "col_b": col_b,
        "reason": f"LLM-recommended {itype} interaction",
    })
    return X


def _apply_bin_quantile(series: pd.Series, q: int = 5) -> pd.Series:
    """Quantile-based discretisation — handles non-linear numeric relationships."""
    try:
        return pd.qcut(series, q=q, labels=False, duplicates="drop").astype(float)
    except Exception:
        return series


# ─────────────────────────────────────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessed_dataset(state):
    raw_data = state.get("raw_data")
    if raw_data is None:
        raise ValueError("Preprocessing expected 'raw_data', but no dataset is loaded.")

    df = raw_data.copy()
    target = state["target_column"]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")

    profiling = state.get("profiling_report", {})
    eda       = state.get("eda_report", {})
    stats     = state.get("stats_report", {})

    # ── Pull LLM decisions from decision_log ──────────────────────────────────
    decision_log    = state.get("decision_log", {})
    pre_decision    = decision_log.get("preprocessing", {})
    llm_fe          = pre_decision.get("feature_engineering", {})   # {col: {action, method, reason}}
    llm_interactions = pre_decision.get("interaction_features", []) # [{col_a, col_b, type, reason}]
    llm_drop        = set(pre_decision.get("drop_features", []))

    column_summary  = profiling.get("column_summary", {})
    recommendations = profiling.get("recommendations", {})
    feature_summary = eda.get("feature_summary", {})
    target_corr     = _target_strength_lookup(eda)
    stats_rank      = _rank_lookup(stats)
    significant_fdr = set(stats.get("significant_after_fdr", []))
    medium_or_large = set(stats.get("medium_effect_features", [])) | \
                      set(stats.get("large_effect_features", []))

    steps         = []
    dropped_cols  = []
    engineered    = []
    transformed   = []
    encoded_cols  = {"frequency": [], "one_hot": [], "target_encode": []}
    imputation    = {"numeric_median": [], "categorical_mode": []}
    fe_steps      = []    # detailed log of every FE step applied

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Drop clearly bad columns
    # ─────────────────────────────────────────────────────────────────────────
    drop_candidates       = set(recommendations.get("drop_candidates", []))
    high_missing_cands    = set(recommendations.get("high_missing", []))

    for col in list(df.columns):
        if col == target:
            continue
        info          = column_summary.get(col, {})
        missing_pct   = info.get("missing_pct", 0)
        corr_strength = target_corr.get(col, 0.0)
        has_signal    = col in significant_fdr or col in medium_or_large or corr_strength >= 0.15

        llm_action = llm_fe.get(col, {}).get("action")

        if llm_action == "drop" or col in llm_drop:
            dropped_cols.append(col)
            fe_steps.append({"col": col, "action": "drop", "method": None,
                              "reason": llm_fe.get(col, {}).get("reason", "LLM-recommended drop")})
        elif col in drop_candidates:
            dropped_cols.append(col)
        elif missing_pct >= 80:
            dropped_cols.append(col)
        elif col in high_missing_cands and not has_signal:
            dropped_cols.append(col)

    if dropped_cols:
        df = df.drop(columns=sorted(set(dropped_cols)), errors="ignore")
        steps.append("dropped_constant_id_or_high_missing_columns")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Datetime feature engineering
    # ─────────────────────────────────────────────────────────────────────────
    for col in list(df.columns):
        if col == target:
            continue
        info     = column_summary.get(col, {})
        sem_type = info.get("semantic_type", "")
        llm_dec  = llm_fe.get(col, {})

        if sem_type not in {"datetime", "datetime_string"}:
            continue

        # Always use cyclical encoding for datetime (better than basic decompose)
        df, new_cols = _apply_cyclical_encoding(df, col, fe_steps)
        engineered.extend(new_cols)

    if engineered:
        steps.append("engineered_datetime_features_with_cyclical_encoding")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Split features / target; impute missing values
    # ─────────────────────────────────────────────────────────────────────────
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    for col in num_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
            imputation["numeric_median"].append(col)

    for col in cat_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(_safe_mode(X[col]))
            imputation["categorical_mode"].append(col)

    if imputation["numeric_median"] or imputation["categorical_mode"]:
        steps.append("imputed_missing_values")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Per-column transforms (LLM-directed first, then rule-based fallback)
    # ─────────────────────────────────────────────────────────────────────────
    outlier_cands = set(recommendations.get("outlier_columns", []))
    skew_cands    = set(recommendations.get("skewed_columns", []))

    for col in [c for c in num_cols if c in X.columns]:
        info     = column_summary.get(col, {})
        eda_info = feature_summary.get(col, {})
        skew_val = info.get("skewness", eda_info.get("skew", 0))
        neg_cnt  = info.get("negatives_count", 0)
        out_pct  = max(info.get("iqr_outlier_pct", 0), eda_info.get("outlier_pct", 0))

        llm_dec  = llm_fe.get(col, {})
        llm_act  = llm_dec.get("action")
        llm_meth = llm_dec.get("method")

        applied = False

        # LLM-directed numeric transforms
        if llm_act == "transform":
            if llm_meth == "log1p" and (X[col].dropna() > -1).all():
                X[col] = np.log1p(X[col])
                transformed.append(f"{col}:log1p")
                fe_steps.append({"col": col, "action": "transform", "method": "log1p",
                                  "reason": llm_dec.get("reason", "LLM: log1p transform")})
                applied = True
            elif llm_meth == "yeo_johnson":
                X[col] = _apply_yeo_johnson(X[col])
                transformed.append(f"{col}:yeo_johnson")
                fe_steps.append({"col": col, "action": "transform", "method": "yeo_johnson",
                                  "reason": llm_dec.get("reason", "LLM: Yeo-Johnson transform")})
                applied = True
            elif llm_meth == "winsorize":
                clean = X[col].dropna()
                if len(clean) >= 4:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        X[col] = X[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
                        transformed.append(f"{col}:winsorized")
                        fe_steps.append({"col": col, "action": "transform", "method": "winsorize",
                                          "reason": llm_dec.get("reason", "LLM: winsorize outliers")})
                        applied = True
            elif llm_meth == "bin_quantile":
                X[col] = _apply_bin_quantile(X[col])
                transformed.append(f"{col}:bin_quantile")
                fe_steps.append({"col": col, "action": "transform", "method": "bin_quantile",
                                  "reason": llm_dec.get("reason", "LLM: quantile binning for non-linear feature")})
                applied = True
            elif llm_meth == "sqrt" and (X[col].dropna() >= 0).all():
                X[col] = np.sqrt(X[col])
                transformed.append(f"{col}:sqrt")
                fe_steps.append({"col": col, "action": "transform", "method": "sqrt",
                                  "reason": llm_dec.get("reason", "LLM: sqrt transform")})
                applied = True

        # Rule-based fallback if LLM didn't specify a transform
        if not applied:
            # Winsorize if heavy outliers
            if col in outlier_cands or out_pct > 5:
                clean = X[col].dropna()
                if len(clean) >= 4:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        X[col] = X[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
                        transformed.append(f"{col}:winsorized")

            # Skew transform
            if (col in skew_cands or abs(skew_val) > 1):
                if neg_cnt == 0 and (X[col].dropna() > -1).all():
                    X[col] = np.log1p(X[col])
                    transformed.append(f"{col}:log1p")
                elif neg_cnt > 0:
                    X[col] = _apply_yeo_johnson(X[col])
                    transformed.append(f"{col}:yeo_johnson")

    if transformed:
        steps.append("applied_numeric_transforms")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Interaction features (LLM-directed)
    # ─────────────────────────────────────────────────────────────────────────
    for spec in llm_interactions:
        col_a = spec.get("col_a")
        col_b = spec.get("col_b")
        itype = spec.get("type", "ratio")
        if col_a and col_b:
            X = _apply_interaction(X, col_a, col_b, itype, fe_steps)
            engineered.append(f"{col_a}_{itype}_{col_b}")

    if llm_interactions:
        steps.append("created_interaction_features")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Drop low-signal features
    # ─────────────────────────────────────────────────────────────────────────
    if significant_fdr:
        drop_low = []
        for col in list(X.columns):
            stats_row     = stats_rank.get(col, {})
            corr_strength = target_corr.get(col, 0.0)
            effect_size   = stats_row.get("effect_size", 0.0)
            mi_score      = stats_row.get("mi_score") or 0.0
            if (
                col not in significant_fdr
                and corr_strength < 0.05
                and effect_size < 0.1
                and mi_score < 0.02
                and col not in medium_or_large
            ):
                drop_low.append(col)

        if drop_low:
            X = X.drop(columns=sorted(set(drop_low)), errors="ignore")
            dropped_cols.extend(drop_low)
            steps.append("dropped_low_signal_features")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Resolve multicollinearity
    # ─────────────────────────────────────────────────────────────────────────
    corr_pairs  = eda.get("correlation", {}).get("high_collinear_pairs", []) or []
    stats_pairs = stats.get("multicollinearity_flags", []) or []

    drop_collinear = set()

    def feature_priority(fn):
        return (
            int(fn in significant_fdr),
            float(target_corr.get(fn, 0.0)),
            float(stats_rank.get(fn, {}).get("effect_size", 0.0)),
        )

    for c1, c2, _ in corr_pairs:
        if c1 in X.columns and c2 in X.columns:
            drop_collinear.add(c2 if feature_priority(c1) >= feature_priority(c2) else c1)

    for flag in stats_pairs:
        c1 = flag.get("col_a")
        c2 = flag.get("col_b")
        if c1 in X.columns and c2 in X.columns:
            drop_collinear.add(c2 if feature_priority(c1) >= feature_priority(c2) else c1)

    if drop_collinear:
        X = X.drop(columns=sorted(drop_collinear), errors="ignore")
        dropped_cols.extend(sorted(drop_collinear))
        steps.append("reduced_multicollinearity")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Encode categorical features
    # ─────────────────────────────────────────────────────────────────────────
    cat_cols     = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    high_card    = set(recommendations.get("high_cardinality", []))
    te_candidates = set(stats.get("fe_signals", {}).get("target_encoding_candidates", []))

    for col in list(cat_cols):
        if col not in X.columns:
            continue

        cardinality  = int(X[col].nunique(dropna=True))
        llm_dec      = llm_fe.get(col, {})
        llm_act      = llm_dec.get("action")
        llm_meth     = llm_dec.get("method")

        use_target_encode = (
            llm_meth == "target_encode"
            or (col in te_candidates and (col in high_card or cardinality > 20))
        )

        if use_target_encode:
            try:
                X[f"{col}_te"] = _apply_target_encoding(X, y, col)
                X = X.drop(columns=[col])
                engineered.append(f"{col}_te")
                encoded_cols["target_encode"].append(col)
                fe_steps.append({"col": col, "action": "encode", "method": "target_encode",
                                  "reason": llm_dec.get("reason", "5-fold target mean encoding")})
            except Exception:
                # Fall back to frequency encoding if target encoding fails
                freqs = X[col].value_counts(normalize=True)
                X[f"{col}_freq"] = X[col].map(freqs).fillna(0.0)
                X = X.drop(columns=[col])
                engineered.append(f"{col}_freq")
                encoded_cols["frequency"].append(col)

        elif col in high_card or cardinality > 20:
            freqs = X[col].value_counts(normalize=True)
            X[f"{col}_freq"] = X[col].map(freqs).fillna(0.0)
            X = X.drop(columns=[col])
            engineered.append(f"{col}_freq")
            encoded_cols["frequency"].append(col)
            fe_steps.append({"col": col, "action": "encode", "method": "frequency",
                              "reason": "high cardinality — frequency encoding"})

    remaining_cat = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if remaining_cat:
        X = pd.get_dummies(X, columns=remaining_cat, drop_first=True)
        encoded_cols["one_hot"].extend(remaining_cat)
        for c in remaining_cat:
            fe_steps.append({"col": c, "action": "encode", "method": "one_hot",
                              "reason": "low cardinality — one-hot encoding"})

    if any(encoded_cols.values()):
        steps.append("encoded_categorical_features")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Scale numeric features
    # ─────────────────────────────────────────────────────────────────────────
    final_num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if final_num_cols:
        scaler = StandardScaler()
        X[final_num_cols] = scaler.fit_transform(X[final_num_cols])
        steps.append("scaled_numeric_features")

    processed_data = pd.concat([X, y], axis=1)
    report = {
        "steps":                 steps,
        "num_features":          len(final_num_cols),
        "cat_features":          len(encoded_cols["one_hot"]) + len(encoded_cols["frequency"]),
        "dropped_columns":       sorted(set(dropped_cols)),
        "engineered_features":   engineered,
        "transformed_columns":   transformed,
        "encoded_columns":       encoded_cols,
        "imputation_summary":    imputation,
        "selected_feature_count": int(X.shape[1]),
        "original_feature_count": int(df.drop(columns=[target]).shape[1]),
        "final_shape":           list(processed_data.shape),
        # Detailed log of every FE step applied — used by notebook_agent
        "fe_steps_applied":      fe_steps,
        "llm_fe_used":           bool(llm_fe),
        "interactions_created":  len(llm_interactions),
    }

    return processed_data, report


def ensure_processed_data(state):
    if "processed_data" in state:
        return state["processed_data"]

    print("⚠️ processed_data missing → auto-generating")
    processed_data, report = build_preprocessed_dataset(state)
    state["processed_data"] = processed_data
    state["preprocessing_report"] = report
    return processed_data
