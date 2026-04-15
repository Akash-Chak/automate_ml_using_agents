# nicegui_app/ui/status_feed.py
#
# One-line-per-agent status panel.
# Each agent completion produces a single concise summary line.
# The tuning line is updated in place (not appended) per trial.

from __future__ import annotations

import time

from nicegui import ui
from nicegui_app.app_state import STEP_LABELS
from nicegui_app.ui.theme import (
    ICON_COLOR_DEFAULT, ICON_COLOR_MAP,
    card_style,
    LABEL_CARD_TITLE, LABEL_STEP_HEADER, LABEL_STEP_TEXT,
    LABEL_ERROR, LABEL_MUTED,
)

_feed_container = None


def _build_feed_line(node_name: str, node_output: dict) -> dict:
    """Extract the most important single-line finding from a node's output."""
    icon = "check_circle"
    text = f"{STEP_LABELS.get(node_name, node_name)} completed."

    if node_name == "profiling":
        report = node_output.get("profiling_report", {})
        n_feat  = report.get("num_columns", "?")
        n_rows  = report.get("num_rows", "?")
        quality = report.get("data_quality_score", {})
        grade   = quality.get("grade", "?")
        score   = quality.get("overall", "?")
        missing = report.get("missing", {}).get("total_missing_pct", 0)
        text = (
            f"{n_feat} features · {n_rows:,} rows · "
            f"quality {grade} ({score}/100) · {missing:.1f}% missing"
        )

    elif node_name == "eda":
        report  = node_output.get("eda_report", {})
        plots   = len(report.get("plots", []))
        insights = report.get("insights", [])
        n_warn  = sum(1 for i in insights if i.startswith("⚠️"))
        corr    = report.get("correlation", {})
        n_strong = len([v for v in corr.get("target_correlations", {}).values() if v > 0.5])
        text = f"{plots} plots generated · {n_strong} strong correlations · {n_warn} warnings"

    elif node_name == "stats":
        report   = node_output.get("stats_report", {})
        n_sig    = len(report.get("significant_after_fdr", []))
        n_total  = report.get("n_features_tested", "?")
        fe       = report.get("fe_signals", {})
        n_nl     = len(fe.get("nonlinear_candidates", []))
        n_large  = len(report.get("large_effect_features", []))
        text = (
            f"{n_sig}/{n_total} significant after FDR · "
            f"{n_large} large effect · "
            f"{n_nl} non-linear candidate{'s' if n_nl != 1 else ''}"
        )

    elif node_name == "decision_pre":
        log     = node_output.get("decision_log", {})
        decision = log.get("preprocessing", {})
        action  = decision.get("action", "proceed")
        fe      = decision.get("feature_engineering", {})
        n_drop  = len(decision.get("drop_features", []))
        n_ix    = len(decision.get("interaction_features", []))
        llm_ok  = decision.get("llm_status") == "success"
        source  = "LLM" if llm_ok else "rule-based"
        fe_methods = list({v.get("method") for v in fe.values() if v.get("method")})[:4]
        fe_str  = ", ".join(fe_methods) if fe_methods else "standard"
        text = f"[{source}] {action} · FE: {fe_str} · {n_drop} drops · {n_ix} interactions"

    elif node_name == "preprocessing":
        report  = node_output.get("preprocessing_report", {})
        dropped = len(report.get("dropped_columns", []))
        trans   = len(report.get("transformed_columns", []))
        enc_f   = len(report.get("encoded_columns", {}).get("frequency", []))
        enc_o   = len(report.get("encoded_columns", {}).get("one_hot", []))
        enc_t   = len(report.get("encoded_columns", {}).get("target_encode", []))
        n_final = report.get("selected_feature_count", "?")
        n_int   = report.get("interactions_created", 0)
        text = (
            f"Dropped {dropped} · transformed {trans} · "
            f"encoded {enc_f + enc_o + enc_t} · "
            f"{n_int} interactions → {n_final} features"
        )

    elif node_name == "baseline":
        result = node_output.get("baseline_result", {})
        model  = result.get("model", "?")
        metric = result.get("metric", "score")
        score  = result.get("score")
        score_str = f"{score:.4f}" if score is not None else "?"
        text = f"Best baseline: {model} · {metric} {score_str}"

    elif node_name == "decision_model":
        log      = node_output.get("decision_log", {})
        decision = log.get("model_selection", {})
        n_models = len(decision.get("candidate_models", []))
        reason   = decision.get("reason", "")[:60]
        llm_ok   = decision.get("llm_status") == "success"
        source   = "LLM" if llm_ok else "rule-based"
        icon = "fork_right"
        text = f"[{source}] {n_models} models selected for tuning · {reason}"

    elif node_name == "tuning":
        result  = node_output.get("advanced_result", {})
        model   = result.get("model", "?")
        metric  = result.get("metric", "score")
        score   = result.get("score")
        cv_score = result.get("tuning_cv_score")
        score_str = f"{score:.4f}" if score is not None else "?"
        cv_str    = f" (CV {cv_score:.4f})" if cv_score is not None else ""
        text = f"Best: {model} · {metric} {score_str}{cv_str}"

    elif node_name == "notebook":
        icon = "auto_stories"
        text = "output_notebook.ipynb ready for download"

    return {"step": node_name, "icon": icon, "text": text}


def _tuning_live_line(event: dict) -> dict | None:
    """Build the live tuning status line from a progress event."""
    phase = event.get("phase")
    if phase == "start":
        return {
            "step": "tuning",
            "icon": "settings",
            "text": (
                f"Tuning {len(event.get('models', []))} models · "
                f"{event.get('optuna_trials_per_model', '?')} trials each · "
                f"metric: {event.get('metric', '?')}"
            ),
        }
    if phase in ("trial_complete", "best_update"):
        label  = event.get("label", "?")
        mi     = event.get("model_index", "?")
        mc     = event.get("model_count", "?")
        ti     = event.get("trial_index", "?")
        tc     = event.get("trial_count", "?")
        score  = event.get("cv_score", "?")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        icon = "trending_up" if phase == "best_update" else "timer"
        return {
            "step": "tuning",
            "icon": icon,
            "text": (
                f"Model {mi}/{mc}: {label} · "
                f"trial {ti}/{tc} · CV {score_str}"
            ),
        }
    if phase == "completed":
        score = event.get("score", "?")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        return {
            "step": "tuning",
            "icon": "check_circle",
            "text": f"Tuning complete · best: {event.get('label', '?')} · {event.get('metric', '')} {score_str}",
        }
    return None


@ui.refreshable
def render():
    state = __import__("nicegui_app.app_state", fromlist=["pipeline_state"]).pipeline_state

    with ui.card().classes("w-full").style(card_style()):
        ui.label("Status").classes(f"{LABEL_CARD_TITLE} mb-3")

        if not state.feed_lines and not state.running:
            ui.label("Run the pipeline to see status updates here.").classes(LABEL_MUTED)
            return

        with ui.column().classes("gap-2 w-full"):
            for line in state.feed_lines:
                _render_line(line)

        if state.error:
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.icon("error").classes("text-red-400 text-lg")
                ui.label(f"Error: {state.error}").classes(LABEL_ERROR)

        if state.completed and not state.error:
            import os
            full_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "output_notebook.ipynb",
            )
            if os.path.exists(full_path):
                with ui.row().classes("mt-3"):
                    ui.button(
                        "Download Notebook",
                        icon="download",
                        on_click=lambda: ui.download(full_path),
                    ).classes("bg-green-700 text-white").style("border-radius:8px;")


def _render_line(line: dict):
    icon       = line.get("icon", "info")
    text       = line.get("text", "")
    step       = line.get("step", "")
    # Icon colour looked up from theme.ICON_COLOR_MAP; default = green
    icon_color = ICON_COLOR_MAP.get(icon, ICON_COLOR_DEFAULT)

    with ui.row().classes("items-start gap-2 w-full"):
        ui.icon(icon).classes(f"{icon_color} text-base mt-0.5")
        with ui.column().classes("gap-0"):
            ui.label(STEP_LABELS.get(step, step)).classes(LABEL_STEP_HEADER)
            ui.label(text).classes(LABEL_STEP_TEXT)


def update_from_node(node_name: str, node_output: dict, do_refresh=True):
    """Called when a pipeline node completes.

    Always updates feed_lines so state stays consistent even if the WebSocket
    is temporarily disconnected.  Pass do_refresh=False to skip the UI repaint
    (caller will trigger a bulk refresh on reconnect instead).
    """
    state = __import__("nicegui_app.app_state", fromlist=["pipeline_state"]).pipeline_state
    line = _build_feed_line(node_name, node_output)
    # Replace existing line for this step or append
    for i, existing in enumerate(state.feed_lines):
        if existing.get("step") == node_name:
            state.feed_lines[i] = line
            if do_refresh:
                render.refresh()
            return
    state.feed_lines.append(line)
    if do_refresh:
        render.refresh()


def tick_running():
    """
    Called every ~1s from the drain loop while the pipeline is running.
    Updates the 'currently running' feed line with live elapsed time,
    giving the user proof that the step is still actively processing.
    """
    state = __import__("nicegui_app.app_state", fromlist=["pipeline_state"]).pipeline_state
    if not state.running:
        return

    # Find the currently-running step
    running_step = next(
        (s for s in state.step_status if state.step_status[s] == "running"), None
    )
    if running_step is None:
        return

    start = state.step_start_times.get(running_step)
    if start is None:
        return

    elapsed = int(time.time() - start)
    mins, secs = divmod(elapsed, 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    # Check if there's already a feed line for this step
    label = STEP_LABELS.get(running_step, running_step)
    running_line = {
        "step": running_step,
        "icon": "pending",
        "text": f"Running… ({elapsed_str} elapsed)",
        "_is_heartbeat": True,
    }

    for i, existing in enumerate(state.feed_lines):
        if existing.get("step") == running_step:
            # Only overwrite if it's our own heartbeat line (don't clobber real output)
            if existing.get("_is_heartbeat"):
                state.feed_lines[i] = running_line
                render.refresh()
            return

    # No line yet — insert heartbeat
    state.feed_lines.append(running_line)
    render.refresh()


def update_from_tuning(event: dict, do_refresh=True):
    """Called per tuning trial — updates the tuning line in place."""
    state = __import__("nicegui_app.app_state", fromlist=["pipeline_state"]).pipeline_state
    line = _tuning_live_line(event)
    if line is None:
        return
    for i, existing in enumerate(state.feed_lines):
        if existing.get("step") == "tuning":
            state.feed_lines[i] = line
            if do_refresh:
                render.refresh()
            return
    state.feed_lines.append(line)
    if do_refresh:
        render.refresh()
