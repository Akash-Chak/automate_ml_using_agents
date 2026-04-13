# Agentic ML Notebook Generator

> Drop any CSV, get back a production-ready Jupyter notebook — a Kaggle-quality starter for any data science project.

**Version 1.0**

---

## What This Does

Point the pipeline at a CSV file with a target column and it runs a full multi-agent analysis. At the end it writes a self-contained `output_notebook.ipynb` that you can open, run top-to-bottom, and submit to Kaggle — or use as the starting point for any new project.

The notebook captures everything the agents discovered: which features matter, how the data was cleaned, what the best model was, and the tuned hyperparameters — all translated into clean, runnable cells.

---

## How It Works

The pipeline is built with **LangGraph**. Each agent is a node in a directed graph that passes a shared state object downstream. An LLM-powered decision agent reads the upstream analysis at two key checkpoints and makes routing decisions — which preprocessing to apply, which model families to try. If the LLM is unavailable it falls back to a deterministic rule-based policy.

```
profiling → eda → stats → decision_pre → preprocessing → baseline → decision_model → tuning → notebook
                                                                             ↓ (if baseline > threshold)
                                                                           notebook (skip tuning)
```

---

## Agents

### 1. Profiling Agent
Deep statistical profiling of every column before any modelling begins.

- **Semantic type inference** — goes beyond dtype to detect: `id`, `datetime`, `boolean`, `binary`, `ordinal`, `categorical`, `continuous`, `text`, `constant`
- **Per-column numeric stats** — mean, median, std, variance, skewness, kurtosis, IQR, coefficient of variation, zero count, negative count
- **Dual outlier detection** — IQR method and Z-score method, both reported
- **Normality testing** — Shapiro-Wilk (≤ 2000 rows) or D'Agostino-Pearson (larger datasets)
- **Distribution shape labelling** — normal-like, right-skewed, left-skewed, leptokurtic, platykurtic
- **Categorical profiling** — top-5 values, Shannon entropy, dominance ratio, cardinality ratio
- **Datetime profiling** — range, unique date count
- **Missing value analysis** — per-column counts, co-missing pairs (columns missing together)
- **Duplicate detection** — row-level and column-level
- **High-multicollinearity flagging** — Pearson |r| > 0.85 pairs
- **Data quality score (0–100)** — composite across completeness, uniqueness, consistency, validity with letter grade (A/B/C/D)
- **Actionable warnings** — drop candidates, high-missing columns, skewed columns, high-cardinality categoricals, near-constant columns
- **Downstream recommendations** — structured hints consumed by later agents

### 2. EDA Agent
Generates a full suite of visualisations saved to `outputs/eda/`.

- **Target distribution** — class bar chart (classification) or histogram with KDE (regression); flags imbalance ratio
- **Per-feature analysis**:
  - Numeric: histogram + KDE with mean/median lines, boxplot with outlier %, scatter vs. target (regression) or grouped boxplot (classification)
  - Categorical: top-10 value bar chart, stacked class distribution (classification) or target boxplot by category (regression)
- **Correlation heatmap** — full lower-triangle Pearson heatmap
- **Target correlation bar chart** — feature correlations ranked, threshold line at |r| = 0.5
- **Missing value chart** — per-column bar chart with 20% threshold line
- **Insight messages** — skew warnings, outlier flags, normality failures, imbalance alerts, multicollinearity warnings, strong/weak feature callouts

### 3. Stats Agent
Rigorous statistical significance testing with effect sizes.

Automatically selects the correct test based on data types and normality/variance assumptions:

| Feature type | Target type | Test chosen |
|---|---|---|
| Numeric | Classification (2 groups, normal + equal var) | Student's t-test |
| Numeric | Classification (2 groups, non-normal or unequal var) | Welch's t or Mann-Whitney U |
| Numeric | Classification (k groups, normal + equal var) | One-way ANOVA |
| Numeric | Classification (k groups, otherwise) | Kruskal-Wallis |
| Numeric | Regression (both normal) | Pearson r |
| Numeric | Regression (non-normal) | Spearman ρ |
| Categorical | Classification | Chi-square + Cramér's V |
| Categorical | Regression (normal + equal var) | ANOVA |
| Categorical | Regression (otherwise) | Kruskal-Wallis |
| Binary numeric | Regression | Point-biserial r |

- **Effect sizes** — Cohen's d, η², ε², Cramér's V, Pearson r, Spearman ρ, point-biserial r — labelled negligible / small / medium / large
- **FDR correction** — Benjamini-Hochberg applied across all p-values; features ranked by significance-after-correction then effect size
- **Multicollinearity scan** — Spearman |r| > 0.85 across all numeric feature pairs
- **Chi-square assumption check** — flags when > 20% of expected cells are < 5

### 4. Decision Agent (LLM-powered)
Runs twice in the pipeline — once before preprocessing and once before model selection. Uses **GPT-4o-mini** to read the upstream reports and decide:

- Which preprocessing steps to apply
- Whether to skip preprocessing entirely
- Which model families to include in the HPO search
- CV folds and Optuna trial budget
- Whether to use class-weight balancing

Falls back to a deterministic rule-based policy if the API is unavailable or returns unparseable output.

### 5. Preprocessing Agent
Executes the preprocessing plan using a shared `build_preprocessed_dataset` utility:

- Drops constant, ID, and high-missing columns flagged by the profiling agent
- Log1p transforms right-skewed numeric columns
- Winsorizes outlier-heavy columns
- Frequency-encodes high-cardinality categoricals
- One-hot encodes low-cardinality categoricals
- Imputes remaining missing values
- Reports every transformation applied

### 6. Baseline Model Agent
Trains a set of lightweight models in seconds to establish a reliable floor.

**Classification candidates:** DummyClassifier, Logistic Regression, Ridge Classifier, Decision Tree

**Regression candidates:** DummyRegressor, Linear Regression, Ridge, Lasso, Decision Tree

- **Imbalance handling** — if minority/majority ratio < 0.35, tests both `class_weight=None` and `class_weight='balanced'` variants; picks the strategy that scores better on weighted F1
- **Primary metric selection** — uses weighted F1 for imbalanced targets, accuracy otherwise; uses R² for regression
- **MLflow logging** — baseline run logged with metrics, params, and a clickable experiment URL

### 7. Hyperparameter Tuning Agent (Advanced Model Agent)
Runs Optuna-based Bayesian optimisation across up to **15+ model families** with cross-validated scoring.

**Classification families:** Logistic Regression, Ridge Classifier, Linear SVC, SVC (RBF), KNN, Gaussian NB, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, Bagging, MLP, XGBoost*, LightGBM*

**Regression families:** Linear Regression, Ridge, Lasso, ElasticNet, Linear SVR, SVR (RBF), KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, Bagging, MLP, XGBoost*, LightGBM*

*If installed

**Three tuning modes** (set via Streamlit sidebar or state):

| Mode | Trials per model | Use case |
|---|---|---|
| `smoke_test` | 3 | Quick sanity check, seconds per model |
| `reuse_mlflow` | — | Reuses best params from a previous MLflow run, skips search |
| `full_search` | Configurable | Full Optuna search for production use |

- Scale-aware — linear/SVM models run inside a `StandardScaler` Pipeline automatically
- Class-weight variants tested for imbalanced classification
- Best params, best score, and all trial results logged to MLflow as child runs under a parent HPO run
- Progress callback support for real-time Streamlit updates

### 8. Notebook Agent
Converts the entire pipeline state into a clean, self-contained Jupyter notebook.

The generated notebook includes:
- Markdown headers and section descriptions written from agent insights
- Import cell with all libraries used in the analysis
- Data loading cell
- EDA section — top feature plots, target distribution, missing value chart
- Preprocessing section — drop list, log transforms, winsorization, encoding — all with exact column names filled in from the pipeline
- Baseline model cell — best model class, training code, evaluation metrics
- Hyperparameter tuning cell — best model with tuned params, cross-validated evaluation
- Feature importance cell (where applicable)
- Evaluation summary with final metrics
- Submission cell template

---

## Pipeline Flow in Detail

```
Input: dataset_path, target_column, problem_type, objective
  │
  ▼
Profiling Agent ──► data quality score, warnings, recommendations
  │
  ▼
EDA Agent ──► plots, insights, correlation summary
  │
  ▼
Stats Agent ──► ranked feature table, significant features, effect sizes
  │
  ▼
Decision Agent (preprocessing) ──► preprocessing plan, candidate models
  │
  ├─ action = "proceed" ──► Preprocessing Agent
  └─ action = "skip"    ──► Baseline directly
                               │
                               ▼
                         Baseline Model Agent ──► best baseline model + score
                               │
                               ▼
                         Decision Agent (model_selection)
                               │
                    ┌──────────┴──────────┐
                    │                     │
              score > threshold      score ≤ threshold
                    │                     │
                    ▼                     ▼
              Notebook Agent      Tuning Agent (Optuna)
                                          │
                                          ▼
                                    Notebook Agent
                                          │
                                          ▼
                                  output_notebook.ipynb
```

---

## Streamlit UI

Run the app for a point-and-click interface:

```bash
streamlit run streamlit_app.py
```

The UI provides:
- CSV upload or local file path input
- Target column and problem type selectors
- Tuning mode selector (smoke test / reuse MLflow / full search)
- Real-time per-agent progress updates via callback injection
- Results panels: data quality score, EDA insights, feature ranking table, baseline metrics, tuned model metrics
- Clickable MLflow experiment URL

---

## MLflow Experiment Tracking

All runs are tracked locally under `./mlruns`. To view:

```bash
mlflow ui --backend-store-uri ./mlruns
# then open http://localhost:5000
```

What is tracked:
- **Baseline run** — model name, accuracy, F1, R², MAE, RMSE, imbalance strategy
- **HPO parent run** — tuning configuration, model family name, number of trials
- **HPO child runs** — one per Optuna trial, with params and score

---

## Project Structure

```
.
├── agents/
│   ├── profiling_agent.py          # Deep column-level statistical profiling
│   ├── eda_agent.py                # Visualisations and insights
│   ├── stats_agent.py              # Hypothesis testing and feature ranking
│   ├── decision_agent.py           # LLM-driven routing decisions
│   ├── preprocessing_agent.py      # Data cleaning and feature engineering
│   ├── baseline_model_agent.py     # Fast baseline model comparison
│   ├── hyperparameter_tuning_agent.py  # Optuna HPO across 15+ model families
│   ├── advanced_model_agent.py     # Delegates to hyperparameter_tuning_agent
│   └── notebook_agent.py           # Generates the output Jupyter notebook
├── orchestrator/
│   └── langgraph_pipeline.py       # LangGraph graph definition and routing logic
├── utils/
│   ├── data_utils.py               # Shared preprocessing helpers
│   └── mlflow_utils.py             # Centralised MLflow logging helpers
├── outputs/
│   └── eda/                        # All plots saved here
├── state.py                        # AgentState TypedDict (shared pipeline state)
├── config.py                       # OpenAI client, API key validation, call_llm()
├── main.py                         # CLI entry point
├── streamlit_app.py                # Streamlit UI
├── output_notebook.ipynb           # Generated notebook (pipeline output)
└── .env                            # API keys (not committed)
```

---

## Setup

**Requirements:** Python 3.10+, conda recommended

```bash
conda activate kaggle

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional, this is the default
```

---

## Running

**Via Streamlit (recommended):**

```bash
streamlit run streamlit_app.py
```

**Via CLI:**

Edit `main.py` with your dataset path, target column, and problem type, then:

```bash
python main.py
```

---

## What the Output Notebook Looks Like

The generated `output_notebook.ipynb` is structured to be:

- **Immediately runnable** — all imports, column names, model classes, and params are filled in from the pipeline's findings
- **Self-documenting** — markdown cells explain what was found and why each step was chosen
- **Kaggle-compatible** — follows the EDA → preprocessing → model → eval → submission structure used in top public Kaggle notebooks
- **A genuine starter, not boilerplate** — the preprocessing code uses the actual columns the pipeline identified as needing treatment; the model cell uses the actual best model with its tuned hyperparameters

---

## Supported Problem Types

| Type | Primary metric | Baseline models | Tuning metric |
|---|---|---|---|
| `classification` | Accuracy or weighted F1 (auto-selected) | Logistic Regression, Ridge, Decision Tree, Dummy | Same as baseline |
| `regression` | R² | Linear Regression, Ridge, Lasso, Decision Tree, Dummy | R² |

---

## Roadmap

Features planned for future versions:

- Feature engineering agent (polynomial features, interaction terms, target encoding)
- Time series problem type support
- Multi-label classification support
- SHAP-based feature importance cells in the output notebook
- Automatic Kaggle submission cell generation
- Support for additional LLM providers (Anthropic Claude, local models)
- Ensemble / stacking agent

---

*Version 1.0 — actively developed. New agents and notebook enrichments added with each version.*
