# Agentic ML Notebook Generator

> Drop any CSV, get back a production-ready Jupyter notebook — a Kaggle-quality starter for any data science project.

---

## What This Does

Point the pipeline at a CSV file with a target column and it runs a full multi-agent analysis. At the end it writes a self-contained `output_notebook.ipynb` that you can open, run top-to-bottom, and submit to Kaggle — or use as the starting point for any new project.

The notebook captures everything the agents discovered: which features matter, how the data was cleaned, what the best model was, and the tuned hyperparameters — all translated into clean, runnable cells.

---

## How It Works

The pipeline is built with **LangGraph**. Each agent is a node in a directed graph that passes a shared state object downstream. An LLM-powered decision agent reads the upstream analysis at two key checkpoints and makes routing decisions. If the LLM is unavailable it falls back to a deterministic rule-based policy.

```
profiling → eda → stats → decision_pre → preprocessing → baseline → decision_model → tuning → notebook
                                                                            ↓ (if baseline > threshold)
                                                                          notebook (skip tuning)
```

---

## Agents

### 1. Profiling Agent
Deep statistical profiling of every column before any modelling begins.

- Semantic type inference — detects: `id`, `datetime`, `boolean`, `binary`, `ordinal`, `categorical`, `continuous`, `text`, `constant`
- Per-column numeric stats — mean, median, std, skewness, kurtosis, IQR, outliers (IQR + Z-score), normality tests
- Categorical profiling — top-5 values, Shannon entropy, dominance ratio, cardinality ratio
- Missing value analysis — per-column counts, co-missing pairs
- Duplicate detection, high-multicollinearity flagging (Pearson |r| > 0.85)
- Data quality score (0–100) with letter grade
- Actionable warnings and downstream recommendations consumed by later agents

### 2. EDA Agent
Generates a full suite of visualisations saved to `outputs/eda/`.

- Target distribution, per-feature histograms/boxplots/scatter plots
- Correlation heatmap, target correlation bar chart, missing value chart
- Insight messages — skew warnings, outlier flags, imbalance alerts, strong/weak feature callouts

### 3. Stats Agent
Rigorous statistical significance testing with effect sizes. Automatically selects the correct test (t-test, Welch's, Mann-Whitney, ANOVA, Kruskal-Wallis, Pearson/Spearman, Chi-square) based on data types and normality assumptions. Applies Benjamini-Hochberg FDR correction and ranks features by significance and effect size.

### 4. Decision Agent (LLM-powered)
Runs twice — once before preprocessing and once before model selection. Uses **GPT-4o-mini** to read upstream reports and decide:

- Which feature engineering actions to apply per column (transform, encode, decompose, drop, keep)
- Which interaction features are worth creating
- Which model families to include in the Optuna search
- CV folds and trial budget

Falls back to a deterministic rule-based policy if the API is unavailable or returns unparseable output. Every prompt sent to the LLM and every response received is logged to `logs/llm_calls_<date>.jsonl`.

### 5. Preprocessing Agent
Executes the decision agent's feature engineering plan:

- Drops constant, ID, and high-missing columns
- Applies log1p, yeo-johnson, winsorize, bin-quantile, sqrt transforms
- Target-encodes high-cardinality categoricals, one-hot encodes low-cardinality
- Decomposes datetime columns (cyclical encoding for month/day-of-week)
- Creates interaction features (ratios, products, differences)
- Imputes remaining missing values

### 6. Baseline Model Agent
Trains lightweight models to establish a performance floor. Tests both unweighted and `class_weight='balanced'` variants for imbalanced targets and picks the better strategy. Logs results to MLflow.

**Classification:** DummyClassifier, Logistic Regression, Ridge Classifier, Decision Tree

**Regression:** DummyRegressor, Linear Regression, Ridge, Lasso, Decision Tree

### 7. Hyperparameter Tuning Agent
Optuna-based Bayesian optimisation across up to 15+ model families with cross-validated scoring. The decision agent selects which families to run; search ranges are fixed per model family.

**Classification families:** Logistic Regression, Ridge Classifier, Linear SVC, SVC (RBF), KNN, Gaussian NB, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, Bagging, MLP, XGBoost*, LightGBM*

**Regression families:** same families, regressor variants + ElasticNet, Linear SVR, SVR (RBF)

*If installed. Best params, best score, and all trial results logged to MLflow.

### 8. Notebook Agent
Converts the entire pipeline state into a clean, self-contained Jupyter notebook:

- Markdown headers written from agent insights
- Data loading, EDA plots, preprocessing steps (exact column names filled in)
- Baseline model evaluation cell
- Optuna hyperparameter tuning cell — runs fresh in the notebook using the same model families and trial budget the agent used
- Feature importance, evaluation summary, submission template

---

## Pipeline Flow

```
Input: dataset_path, target_column, problem_type
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
Decision Agent (preprocessing) ──► feature engineering plan
  │
  ├─ action = "proceed" ──► Preprocessing Agent
  └─ action = "skip"    ──► Baseline directly
                               │
                               ▼
                         Baseline Model Agent ──► best baseline model + score
                               │
                               ▼
                         Decision Agent (model_selection) ──► candidate models
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

## UI

Run the NiceGUI app for a point-and-click interface:

```bash
python -m nicegui_app.main
# then open http://localhost:8080
```

The UI provides:
- CSV upload or local file path input
- Target column and problem type selectors
- Real-time per-agent progress updates
- Results panels: data quality score, EDA insights, feature ranking, baseline and tuned model metrics
- Integrated MLflow experiment viewer

---

## MLflow Experiment Tracking

All runs are tracked locally under `./mlruns`. To view:

```bash
mlflow ui --backend-store-uri ./mlruns
# then open http://localhost:5000
```

Tracked per run:
- **Baseline run** — model name, accuracy, F1, R², MAE, RMSE, imbalance strategy
- **HPO parent run** — tuning config, model family, number of trials
- **HPO child runs** — one per Optuna trial with params and score

---

## LLM Call Logs

Every prompt sent to OpenAI and every response received is written to `logs/llm_calls_<date>.jsonl` (one JSON object per line). Fields: `timestamp`, `agent`, `model`, `prompt`, `response`, `usage` (token counts).

```python
import pandas as pd
df = pd.read_json("logs/llm_calls_2026-04-15.jsonl", lines=True)
```

---

## Project Structure

```
.
├── agents/
│   ├── profiling_agent.py
│   ├── eda_agent.py
│   ├── stats_agent.py
│   ├── decision_agent.py
│   ├── preprocessing_agent.py
│   ├── baseline_model_agent.py
│   ├── hyperparameter_tuning_agent.py
│   ├── advanced_model_agent.py
│   └── notebook_agent.py
├── nicegui_app/
│   ├── main.py                  # UI entry point (python -m nicegui_app.main)
│   ├── pipeline_runner.py       # Runs the LangGraph pipeline in a background thread
│   ├── app_state.py             # Shared UI state
│   ├── mlflow_server.py         # Embedded MLflow server helper
│   └── ui/
│       ├── layout.py
│       ├── sidebar.py
│       ├── flow_diagram.py
│       └── status_feed.py
├── orchestrator/
│   └── langgraph_pipeline.py
├── utils/
│   ├── data_utils.py
│   ├── mlflow_utils.py
│   └── markdown_formatter.py
├── logs/                        # LLM call logs (llm_calls_<date>.jsonl)
├── outputs/eda/                 # EDA plots
├── state.py
├── config.py                    # OpenAI client, call_llm(), LLM logger
├── main.py                      # CLI entry point
└── .env                         # API keys (not committed)
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

**Via NiceGUI (recommended):**

```bash
python -m nicegui_app.main
```

**Via CLI:**

Edit `main.py` with your dataset path, target column, and problem type, then:

```bash
python main.py
```
