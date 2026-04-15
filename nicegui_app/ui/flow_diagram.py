# nicegui_app/ui/flow_diagram.py
#
# Live ECharts directed graph showing the agent pipeline.
# Nodes light up as each agent runs; branch edges resolve when
# decision nodes complete.
#
# ── To change colours / sizes: edit nicegui_app/ui/theme.py ──────────────────
# ── To change node positions / labels: edit the CONFIG block below ───────────

from __future__ import annotations
from nicegui import ui
from nicegui_app.app_state import STEPS, STEP_LABELS
from nicegui_app.ui.theme import (
    NODE_FILL, NODE_TEXT,
    NODE_GLOW_COLOR, NODE_GLOW_BLUR,
    EDGE_DONE, EDGE_PENDING, EDGE_BRANCH_TAKEN, EDGE_BRANCH_SKIP,
    LEGEND_ITEMS,
    DIAGRAM_NODE_SIZE, DIAGRAM_FONT_SIZE,
    DIAGRAM_CHART_HEIGHT, DIAGRAM_CURVENESS, DIAGRAM_EDGE_ARROW,
    card_style, LABEL_CARD_TITLE,
)

# ══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM CONFIG — edit these two dicts to change layout and labels
# ══════════════════════════════════════════════════════════════════════════════

# Short labels that fit inside the node boxes.
# Full labels (used in the status feed) live in app_state.STEP_LABELS.
DIAGRAM_LABELS: dict[str, str] = {
    "profiling":      "Profiling",
    "eda":            "EDA",
    "stats":          "Statistics",
    "decision_pre":   "Decision\n(Pre)",
    "preprocessing":  "Preprocess",
    "baseline":       "Baseline",
    "decision_model": "Decision\n(Model)",
    "tuning":         "HP Tuning",
    "notebook":       "Notebook",
}

# (x, y) pixel positions in ECharts coordinate space.
# Horizontal step = 185 px, node width = 110 px → 75 px gap between edges.
# Main row y=240; fork-up (preprocessing) y=120; fork-down (tuning) y=360.
NODE_POS: dict[str, tuple[int, int]] = {
    "profiling":      (80,   240),
    "eda":            (265,  240),
    "stats":          (450,  240),
    "decision_pre":   (635,  240),
    "preprocessing":  (820,  120),   # branch A — above main row
    "baseline":       (1005, 240),
    "decision_model": (1190, 240),
    "tuning":         (1375, 360),   # branch A — below main row
    "notebook":       (1560, 240),
}

# ══════════════════════════════════════════════════════════════════════════════

# Edges: (source, target, is_conditional)
# is_conditional=True → dashed until branch_taken resolves which edge was taken
EDGES = [
    ("profiling",      "eda",            False),
    ("eda",            "stats",          False),
    ("stats",          "decision_pre",   False),
    ("decision_pre",   "preprocessing",  True),   # branch A
    ("decision_pre",   "baseline",       True),   # branch B (skip preprocessing)
    ("preprocessing",  "baseline",       False),
    ("baseline",       "decision_model", False),
    ("decision_model", "tuning",         True),   # branch A
    ("decision_model", "notebook",       True),   # branch B (skip tuning)
    ("tuning",         "notebook",       False),
]

# Global reference to the echart element so update() can push diffs
_echart_ref = None


def _build_options(step_status: dict, branch_taken: dict, trial_badge: int = 0) -> dict:
    nodes = []
    for step in STEPS:
        status = step_status.get(step, "pending")
        x, y   = NODE_POS[step]
        label  = DIAGRAM_LABELS.get(step, step)

        # Tuning node: append live trial count while running
        if step == "tuning" and status == "running" and trial_badge > 0:
            label = f"HP Tuning\n({trial_badge} trials)"

        # Extra style for the running tuning node (border highlight)
        extra: dict = {}
        if step == "tuning" and status == "running":
            extra = {
                "itemStyle": {
                    "color":       NODE_FILL[status],
                    "borderColor": NODE_GLOW_COLOR,
                    "borderWidth": 2,
                },
                "emphasis": {"scale": True},
            }

        nodes.append({
            "name":       step,
            "x":          x,
            "y":          y,
            "symbol":     "roundRect",
            "symbolSize": DIAGRAM_NODE_SIZE,
            "label": {
                "show":       True,
                "formatter":  label,
                "fontSize":   DIAGRAM_FONT_SIZE,
                "fontWeight": "bold" if status == "running" else "normal",
                "color":      NODE_TEXT.get(status, "#e2e8f0"),
            },
            "itemStyle": {
                "color":        NODE_FILL.get(status, NODE_FILL["pending"]),
                "borderRadius": 8,
                "shadowBlur":   NODE_GLOW_BLUR if status == "running" else 0,
                "shadowColor":  NODE_GLOW_COLOR if status == "running" else "transparent",
            },
            **extra,
        })

    links = []
    for src, dst, is_conditional in EDGES:
        if is_conditional:
            taken_dst = branch_taken.get(src)
            if taken_dst is None:
                # Not yet resolved — dashed grey
                line_type  = "dashed"
                line_color = EDGE_PENDING
                line_width = 1
            elif taken_dst == dst:
                # This branch was taken — solid green
                line_type  = "solid"
                line_color = EDGE_BRANCH_TAKEN
                line_width = 2
            else:
                # Other branch — skipped, dimmed
                line_type  = "dashed"
                line_color = EDGE_BRANCH_SKIP
                line_width = 1
        else:
            src_done   = step_status.get(src, "pending") == "done"
            line_type  = "solid"
            line_color = EDGE_DONE if src_done else EDGE_PENDING
            line_width = 2 if src_done else 1

        links.append({
            "source": src,
            "target": dst,
            "lineStyle": {
                "type":  line_type,
                "color": line_color,
                "width": line_width,
            },
        })

    return {
        "backgroundColor": "transparent",
        "series": [{
            "type":                  "graph",
            "layout":                "none",
            "roam":                  False,
            "coordinateSystem":      None,
            "edgeSymbol":            ["none", "arrow"],
            "edgeSymbolSize":        DIAGRAM_EDGE_ARROW,
            "data":                  nodes,
            "links":                 links,
            "lineStyle":             {"curveness": DIAGRAM_CURVENESS},
            "emphasis":              {"focus": "adjacency"},
            "animation":             True,
            "animationDurationUpdate": 300,
        }],
    }


def render():
    global _echart_ref
    _echart_ref = None
    state = __import__("nicegui_app.app_state", fromlist=["pipeline_state"]).pipeline_state

    with ui.card().classes("w-full").style(card_style()):
        ui.label("Agent Pipeline").classes(f"{LABEL_CARD_TITLE} mb-2")

        # Legend row — colours come from theme.LEGEND_ITEMS
        with ui.row().classes("gap-4 mb-3"):
            for label, color in LEGEND_ITEMS:
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").style(
                        f"width:12px; height:12px; border-radius:3px; background:{color};"
                    )
                    ui.label(label).classes("text-xs text-slate-400")

        _echart_ref = ui.echart(
            _build_options(state.step_status, state.branch_taken)
        ).style(f"height:{DIAGRAM_CHART_HEIGHT}; width:100%;")


def update(step_status: dict, branch_taken: dict, trial_badge: int = 0):
    """Called from the drain loop to push a live diff to the browser.

    Uses run_chart_method('setOption', ..., True) — the notMerge=True flag
    forces ECharts to fully replace node/edge styles rather than merging
    with cached state, which is what causes colour updates to be silently dropped.
    """
    global _echart_ref

    if _echart_ref is None:
        return

    # ✅ Check if client is still alive
    try:
        if _echart_ref.client is None or not _echart_ref.client.connected:
            return
    except Exception:
        return

    # ✅ Safe update (prevents crash if client deleted mid-call)
    try:
        _echart_ref.run_chart_method(
            'setOption',
            _build_options(step_status, branch_taken, trial_badge),
            True,
        )
    except RuntimeError:
        # Client already deleted — ignore safely
        return
