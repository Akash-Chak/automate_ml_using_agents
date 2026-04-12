def format_eda_report(report):
    if not report:
        return "No EDA insights available."

    text = "### Key Insights\n\n"

    for insight in report.get("insights", []):
        text += f"- {insight}\n"

    plots = report.get("plots", [])
    text += "\n### Plots Generated\n\n"

    if plots:
        for plot_path in plots:
            text += f"- {plot_path}\n"
    else:
        text += "- No plots generated\n"

    return text


def format_stats_report(report):
    if not report:
        return "No statistical analysis available."

    text = "### Statistical Summary\n\n"

    sig = report.get("significant", [])
    insig = report.get("insignificant", [])

    text += f"**Significant Features ({len(sig)}):**\n"
    for feature in sig:
        text += f"- {feature}\n"

    text += f"\n**Insignificant Features ({len(insig)}):**\n"
    for feature in insig:
        text += f"- {feature}\n"

    return text


def format_preprocessing(report):
    if not report:
        return "No preprocessing summary available."

    text = "### Preprocessing Steps\n\n"

    for step in report.get("steps", []):
        text += f"- {step}\n"

    text += f"\n**Numerical Features:** {report.get('num_features', 0)}\n"
    text += f"**Categorical Features:** {report.get('cat_features', 0)}\n"
    text += f"**Selected Features:** {report.get('selected_feature_count', 0)}\n"

    dropped = report.get("dropped_columns", [])
    if dropped:
        text += "\n**Dropped Columns:**\n"
        for col in dropped:
            text += f"- {col}\n"

    engineered = report.get("engineered_features", [])
    if engineered:
        text += "\n**Engineered Features:**\n"
        for col in engineered:
            text += f"- {col}\n"

    return text


def format_model_result(report, title="Model"):
    if not report:
        return f"No {title} results."

    text = f"### {title} Results\n\n"

    for key, value in report.items():
        text += f"- **{key}**: {value}\n"

    return text


def format_decision(report):
    text = "### Decision Summary\n\n"

    for stage, details in report.items():
        text += f"#### {stage.upper()}\n"
        text += f"- **Action**: {details.get('action')}\n"
        text += f"- **Reason**: {details.get('reason')}\n"
        text += f"- **Recommended Model**: {details.get('recommended_model')}\n"

        steps = details.get("preprocessing_steps", [])
        if steps:
            text += "- **Suggested Steps:**\n"
            for step in steps:
                text += f"  - {step}\n"

        text += "\n"

    return text
