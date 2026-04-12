# agents/stats_agent.py

from scipy.stats import chi2_contingency, f_oneway

def stats_agent(state):
    df = state["raw_data"]
    target = state["target_column"]

    significant = []
    insignificant = []

    for col in df.columns:
        if col == target:
            continue

        try:
            if df[col].dtype == "object":
                contingency = pd.crosstab(df[col], df[target])
                chi2, p, _, _ = chi2_contingency(contingency)

                if p < 0.05:
                    significant.append(col)
                else:
                    insignificant.append(col)

        except:
            continue

    state["stats_report"] = {
        "significant": significant,
        "insignificant": insignificant
    }

    return state