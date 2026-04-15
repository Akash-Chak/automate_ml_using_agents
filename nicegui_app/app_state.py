# nicegui_app/app_state.py
#
# Single shared mutable state for the pipeline run.
# This is a single-user local tool so module-level state is fine.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STEPS = [
    "profiling",
    "eda",
    "stats",
    "decision_pre",
    "preprocessing",
    "baseline",
    "decision_model",
    "tuning",
    "notebook",
]

STEP_LABELS = {
    "profiling":     "Data Profiling",
    "eda":           "EDA",
    "stats":         "Statistics",
    "decision_pre":  "Decision (Pre)",
    "preprocessing": "Preprocessing",
    "baseline":      "Baseline Model",
    "decision_model": "Decision (Model)",
    "tuning":        "Hyperparameter Tuning",
    "notebook":      "Generate Notebook",
}


@dataclass
class PipelineState:
    # Run lifecycle
    running:    bool = False
    stop:       bool = False
    completed:  bool = False
    error:      Optional[str] = None

    # Per-step state: pending | running | done | skipped | error
    step_status:      Dict[str, str]   = field(default_factory=lambda: {s: "pending" for s in STEPS})
    step_times:       Dict[str, float] = field(default_factory=dict)   # finish timestamps
    step_start_times: Dict[str, float] = field(default_factory=dict)   # start timestamps

    # LangGraph node outputs accumulated here
    latest_state: Dict[str, Any] = field(default_factory=dict)

    # Status feed — one line per agent (updated in place)
    feed_lines: List[Dict] = field(default_factory=list)   # [{step, icon, text}]

    # Live tuning status (updated per trial)
    tuning_live: Dict = field(default_factory=dict)

    # Config from sidebar (populated on Run)
    config: Dict[str, Any] = field(default_factory=dict)

    # Detected columns for the sidebar dropdown
    detected_columns: List[str] = field(default_factory=list)

    # Branch resolution: which edge was taken at each decision node
    # "decision_pre"  → "preprocessing" | "baseline"
    # "decision_model" → "tuning" | "notebook"
    branch_taken: Dict[str, str] = field(default_factory=dict)

    # Async queue for pipeline events
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def reset(self):
        self.running     = False
        self.stop        = False
        self.completed   = False
        self.error       = None
        self.step_status      = {s: "pending" for s in STEPS}
        self.step_times       = {}
        self.step_start_times = {}
        self.latest_state     = {}
        self.feed_lines  = []
        self.tuning_live = {}
        self.branch_taken = {}
        # Don't reset config, detected_columns — user may re-run same dataset


# Module-level singleton
pipeline_state = PipelineState()
