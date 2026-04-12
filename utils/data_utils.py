import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


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


def build_preprocessed_dataset(state):
    raw_data = state.get("raw_data")
    if raw_data is None:
        raise ValueError("Preprocessing expected 'raw_data', but no dataset is loaded.")

    df = raw_data.copy()
    target = state["target_column"]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")

    profiling = state.get("profiling_report", {})
    eda = state.get("eda_report", {})
    stats = state.get("stats_report", {})

    column_summary = profiling.get("column_summary", {})
    recommendations = profiling.get("recommendations", {})
    feature_summary = eda.get("feature_summary", {})
    target_corr = _target_strength_lookup(eda)
    stats_rank = _rank_lookup(stats)
    significant_fdr = set(stats.get("significant_after_fdr", []))
    medium_or_large = set(stats.get("medium_effect_features", [])) | set(stats.get("large_effect_features", []))

    steps = []
    dropped_columns = []
    engineered_features = []
    transformed_columns = []
    encoded_columns = {"frequency": [], "one_hot": []}
    imputation_summary = {"numeric_median": [], "categorical_mode": []}

    # 1. Drop clearly bad columns first.
    drop_candidates = set(recommendations.get("drop_candidates", []))
    high_missing_candidates = set(recommendations.get("high_missing", []))

    for col in list(df.columns):
        if col == target:
            continue

        info = column_summary.get(col, {})
        missing_pct = info.get("missing_pct", 0)
        corr_strength = target_corr.get(col, 0.0)
        keep_due_to_signal = col in significant_fdr or col in medium_or_large or corr_strength >= 0.15

        if col in drop_candidates:
            dropped_columns.append(col)
        elif missing_pct >= 80:
            dropped_columns.append(col)
        elif col in high_missing_candidates and not keep_due_to_signal:
            dropped_columns.append(col)

    if dropped_columns:
        df = df.drop(columns=sorted(set(dropped_columns)), errors="ignore")
        steps.append("dropped_constant_id_or_high_missing_columns")

    # 2. Engineer datetime features from profiling semantics.
    for col in list(df.columns):
        if col == target:
            continue

        info = column_summary.get(col, {})
        sem_type = info.get("semantic_type")
        if sem_type not in {"datetime", "datetime_string"}:
            continue

        series = pd.to_datetime(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue

        new_cols = {
            f"{col}_year": series.dt.year,
            f"{col}_month": series.dt.month,
            f"{col}_day": series.dt.day,
            f"{col}_dayofweek": series.dt.dayofweek,
        }
        for new_col, values in new_cols.items():
            df[new_col] = values
            engineered_features.append(new_col)

        df = df.drop(columns=[col], errors="ignore")
        dropped_columns.append(col)

    if engineered_features:
        steps.append("engineered_datetime_features")

    # 3. Split features and target.
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # 4. Impute missing values using dtype-aware strategy.
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    for col in num_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
            imputation_summary["numeric_median"].append(col)

    for col in cat_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(_safe_mode(X[col]))
            imputation_summary["categorical_mode"].append(col)

    if imputation_summary["numeric_median"] or imputation_summary["categorical_mode"]:
        steps.append("imputed_missing_values")

    # 5. Cap outliers using IQR on flagged numeric columns.
    outlier_candidates = set(recommendations.get("outlier_columns", []))
    for col in [c for c in num_cols if c in X.columns]:
        info = column_summary.get(col, {})
        eda_info = feature_summary.get(col, {})
        outlier_pct = max(info.get("iqr_outlier_pct", 0), eda_info.get("outlier_pct", 0))
        if col in outlier_candidates or outlier_pct > 5:
            clean = X[col].dropna()
            if len(clean) >= 4:
                q1 = clean.quantile(0.25)
                q3 = clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    X[col] = X[col].clip(lower=lower, upper=upper)
                    transformed_columns.append(f"{col}:winsorized")

    # 6. Transform skewed positive numeric columns.
    skew_candidates = set(recommendations.get("skewed_columns", []))
    for col in [c for c in num_cols if c in X.columns]:
        info = column_summary.get(col, {})
        eda_info = feature_summary.get(col, {})
        skew_val = info.get("skewness", eda_info.get("skew", 0))
        if (col in skew_candidates or abs(skew_val) > 1) and (X[col].dropna() > -1).all():
            X[col] = np.log1p(X[col])
            transformed_columns.append(f"{col}:log1p")

    if transformed_columns:
        steps.append("treated_outliers_and_skewness")

    # 7. Drop low-value/insignificant features when stronger evidence exists.
    if significant_fdr:
        drop_low_signal = []
        for col in list(X.columns):
            stats_row = stats_rank.get(col, {})
            corr_strength = target_corr.get(col, 0.0)
            effect_size = stats_row.get("effect_size", 0.0)
            if (
                col not in significant_fdr
                and corr_strength < 0.05
                and effect_size < 0.1
                and col not in medium_or_large
            ):
                drop_low_signal.append(col)

        if drop_low_signal:
            X = X.drop(columns=sorted(set(drop_low_signal)), errors="ignore")
            dropped_columns.extend(drop_low_signal)
            steps.append("dropped_low_signal_features")

    # 8. Resolve multicollinearity by dropping weaker feature from strong pairs.
    corr_pairs = eda.get("correlation", {}).get("high_collinear_pairs", []) or []
    stats_pairs = stats.get("multicollinearity_flags", []) or []

    drop_for_collinearity = set()

    def feature_priority(feature_name: str):
        return (
            int(feature_name in significant_fdr),
            float(target_corr.get(feature_name, 0.0)),
            float(stats_rank.get(feature_name, {}).get("effect_size", 0.0)),
        )

    for c1, c2, _ in corr_pairs:
        if c1 in X.columns and c2 in X.columns:
            drop_for_collinearity.add(c2 if feature_priority(c1) >= feature_priority(c2) else c1)

    for flag in stats_pairs:
        c1 = flag.get("col_a")
        c2 = flag.get("col_b")
        if c1 in X.columns and c2 in X.columns:
            drop_for_collinearity.add(c2 if feature_priority(c1) >= feature_priority(c2) else c1)

    if drop_for_collinearity:
        X = X.drop(columns=sorted(drop_for_collinearity), errors="ignore")
        dropped_columns.extend(sorted(drop_for_collinearity))
        steps.append("reduced_multicollinearity")

    # 9. Encode categorical variables using cardinality-aware strategy.
    cat_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    high_cardinality = set(recommendations.get("high_cardinality", []))

    for col in list(cat_cols):
        if col not in X.columns:
            continue

        cardinality = int(X[col].nunique(dropna=True))
        if col in high_cardinality or cardinality > 20:
            freqs = X[col].value_counts(normalize=True)
            X[f"{col}_freq"] = X[col].map(freqs).fillna(0.0)
            X = X.drop(columns=[col])
            engineered_features.append(f"{col}_freq")
            encoded_columns["frequency"].append(col)

    remaining_cat_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if remaining_cat_cols:
        X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)
        encoded_columns["one_hot"].extend(remaining_cat_cols)

    if encoded_columns["frequency"] or encoded_columns["one_hot"]:
        steps.append("encoded_categorical_features")

    # 10. Scale numeric features for downstream linear models.
    final_num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if final_num_cols:
        scaler = StandardScaler()
        X[final_num_cols] = scaler.fit_transform(X[final_num_cols])
        steps.append("scaled_numeric_features")

    processed_data = pd.concat([X, y], axis=1)
    report = {
        "steps": steps,
        "num_features": len(final_num_cols),
        "cat_features": len(encoded_columns["one_hot"]) + len(encoded_columns["frequency"]),
        "dropped_columns": sorted(set(dropped_columns)),
        "engineered_features": engineered_features,
        "transformed_columns": transformed_columns,
        "encoded_columns": encoded_columns,
        "imputation_summary": imputation_summary,
        "selected_feature_count": int(X.shape[1]),
        "original_feature_count": int(df.drop(columns=[target]).shape[1]),
        "final_shape": list(processed_data.shape),
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
