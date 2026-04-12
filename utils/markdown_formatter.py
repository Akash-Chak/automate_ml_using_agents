# utils/markdown_formatter.py

def format_eda_report(report):
    if not report:
        return "No EDA insights available."

    text = "### 🔍 Key Insights\n\n"

    for insight in report.get("insights", []):
        text += f"- {insight}\n"

    plots = report.get("plots", [])
    text += "\n### 📊 Plots Generated\n\n"

    if plots:
        for p in plots:
            text += f"- {p}\n"
    else:
        text += "- No plots generated\n"

    return text


def format_stats_report(report):
    if not report:
        return "No statistical analysis available."

    text = "### 📈 Statistical Summary\n\n"

    sig = report.get("significant", [])
    insig = report.get("insignificant", [])

    text += f"**Significant Features ({len(sig)}):**\n"
    for s in sig:
        text += f"- {s}\n"

    text += f"\n**Insignificant Features ({len(insig)}):**\n"
    for s in insig:
        text += f"- {s}\n"

    return text


def format_preprocessing(report):
    text = "### 🧹 Preprocessing Steps\n\n"

    for step in report.get("steps", []):
        text += f"- {step}\n"

    text += f"\n**Numerical Features:** {report.get('num_features', 0)}\n"
    text += f"**Categorical Features:** {report.get('cat_features', 0)}\n"

    return text


def format_model_result(report, title="Model"):
    if not report:
        return f"No {title} results."

    text = f"### 🤖 {title} Results\n\n"

    for k, v in report.items():
        text += f"- **{k}**: {v}\n"

    return text


def format_decision(report):
    text = "### 🧠 Decision Summary\n\n"

    for stage, details in report.items():
        text += f"#### {stage.upper()}\n"
        text += f"- **Action**: {details.get('action')}\n"
        text += f"- **Reason**: {details.get('reason')}\n"
        text += f"- **Recommended Model**: {details.get('recommended_model')}\n"

        steps = details.get("preprocessing_steps", [])
        if steps:
            text += "- **Suggested Steps:**\n"
            for s in steps:
                text += f"  - {s}\n"

        text += "\n"

    return text