# nicegui_app/ui/layout.py
#
# Page shell: fixed-width sidebar column + main content area with
# agent flow diagram + status feed.
# Houses the 100ms drain loop that converts queue events into UI updates.
#
# ── To change colours / styles: edit nicegui_app/ui/theme.py ─────────────────
# ── To change page-level CSS:   edit nicegui_app/static/styles.css ───────────

from __future__ import annotations

import asyncio
import time

from nicegui import ui

from nicegui_app.app_state import pipeline_state, STEPS
from nicegui_app.pipeline_runner import run_pipeline
from nicegui_app.ui import flow_diagram, sidebar, status_feed
from nicegui_app.ui.sidebar import set_running as _sidebar_set_running
from nicegui_app.ui.theme import (
    header_style, sidebar_style, page_style,
)

_last_tick_second: float = 0.0   # throttle heartbeat to ~1s
_was_connected = True             # track reconnection to force UI re-sync


# ── Branch resolution ─────────────────────────────────────────────────────────

def _resolve_branch(node_name: str, node_output: dict):
    """
    After a decision node completes, record which downstream edge was taken.
    This drives the conditional edge colouring in the flow diagram.
    """
    state = pipeline_state

    if node_name == "decision_pre":
        log     = node_output.get("decision_log", {})
        action  = log.get("preprocessing", {}).get("action", "proceed")
        state.branch_taken["decision_pre"] = (
            "baseline" if action == "skip_preprocessing" else "preprocessing"
        )

    elif node_name == "decision_model":
        log     = node_output.get("decision_log", {})
        action  = log.get("model_selection", {}).get("action", "tune")
        state.branch_taken["decision_model"] = (
            "notebook" if action == "skip_tuning" else "tuning"
        )


def _mark_skipped_nodes():
    state = pipeline_state
    if state.branch_taken.get("decision_pre") == "baseline":
        if state.step_status.get("preprocessing") == "pending":
            state.step_status["preprocessing"] = "skipped"
    if state.branch_taken.get("decision_model") == "notebook":
        if state.step_status.get("tuning") == "pending":
            state.step_status["tuning"] = "skipped"


def _mark_next_running(just_completed: str):
    """Mark the next expected node as 'running' so the diagram lights up amber
    immediately, before the real node_complete event arrives."""
    state       = pipeline_state
    taken_pre   = state.branch_taken.get("decision_pre")
    taken_model = state.branch_taken.get("decision_model")

    next_map = {
        "profiling":      "eda",
        "eda":            "stats",
        "stats":          "decision_pre",
        "decision_pre":   taken_pre   or "preprocessing",
        "preprocessing":  "baseline",
        "baseline":       "decision_model",
        "decision_model": taken_model or "tuning",
        "tuning":         "notebook",
    }
    nxt = next_map.get(just_completed)
    if nxt and state.step_status.get(nxt) == "pending":
        state.step_status[nxt] = "running"
        state.step_start_times[nxt] = time.time()


# ── Queue drain ───────────────────────────────────────────────────────────────

async def _drain_queue():
    """
    Drain all pending pipeline events from the asyncio.Queue.
    Called directly by ui.timer (NOT wrapped in asyncio.create_task)
    so that NiceGUI's slot context is preserved and ui.notify works.

    State updates (step_status, step_start_times, etc.) always run regardless
    of WebSocket connection state.  UI refresh calls are guarded so they only
    fire when the client is connected — this prevents a brief "trying to
    reconnect" moment from stalling state and losing elapsed-time tracking.
    """
    from nicegui import ui

    client = ui.context.client
    connected = client is not None and client.connected

    global _last_tick_second, _was_connected
    state = pipeline_state
    queue = state.event_queue

    # Re-sync UI when client just reconnected after a dropout
    just_reconnected = connected and not _was_connected
    _was_connected = connected
    if just_reconnected and state.running:
        flow_diagram.update(state.step_status, state.branch_taken,
                            state.tuning_live.get("trial_badge", 0))
        status_feed.render.refresh()

    # Heartbeat: update elapsed-time label once per second (only when connected)
    now = time.time()
    if connected and state.running and now - _last_tick_second >= 1.0:
        _last_tick_second = now
        status_feed.tick_running()

    processed = 0
    while not queue.empty() and processed < 20:
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        processed += 1
        event_type = event[0]

        if event_type == "node_complete":
            _, node_name, node_output = event
            state.step_times[node_name] = time.time()
            state.step_status[node_name] = "done"
            _resolve_branch(node_name, node_output)
            state.latest_state.update(node_output)
            _mark_skipped_nodes()
            _mark_next_running(node_name)
            # Always update feed_lines; only repaint when the socket is live
            status_feed.update_from_node(node_name, node_output, do_refresh=connected)
            if connected:
                flow_diagram.update(
                    state.step_status,
                    state.branch_taken,
                    state.tuning_live.get("trial_badge", 0),
                )
                flow_diagram.update(state.step_status, state.branch_taken)

        elif event_type == "tuning_event":
            _, tuning_event = event
            if tuning_event.get("phase") in ("trial_complete", "best_update"):
                state.tuning_live["trial_badge"] = tuning_event.get("trial_index", 0)
            # Always update tuning feed line so it's ready when socket reconnects
            status_feed.update_from_tuning(tuning_event, do_refresh=connected)
            if connected:
                flow_diagram.update(
                    state.step_status,
                    state.branch_taken,
                    state.tuning_live.get("trial_badge", 0),
                )

        elif event_type == "pipeline_done":
            state.running   = False
            state.completed = True
            state.step_status = {
                s: ("done" if state.step_status.get(s) != "skipped" else "skipped")
                for s in STEPS
            }
            if connected:
                flow_diagram.update(state.step_status, state.branch_taken)
                status_feed.render.refresh()
                _sidebar_set_running(False)
                ui.notify("Pipeline complete!", color="positive", timeout=4000)

        elif event_type == "pipeline_stopped":
            state.running = False
            if connected:
                flow_diagram.update(state.step_status, state.branch_taken)
                status_feed.render.refresh()
                _sidebar_set_running(False)
                ui.notify("Pipeline stopped.", color="warning", timeout=3000)

        elif event_type == "pipeline_error":
            _, error_msg = event
            state.running = False
            state.error   = error_msg
            for s in STEPS:
                if state.step_status.get(s) == "running":
                    state.step_status[s] = "error"
            if connected:
                flow_diagram.update(state.step_status, state.branch_taken)
                status_feed.render.refresh()
                _sidebar_set_running(False)
                ui.notify(f"Error: {error_msg}", color="negative", timeout=0)


# ── Page ─────────────────────────────────────────────────────────────────────

def build_page():
    """
    Construct the single-page UI.
    Called from the @ui.page("/") handler in main.py.
    """
    state = pipeline_state

    async def handle_run(config: dict):
        # Lock sidebar immediately so a second Run can't fire mid-pipeline.
        _sidebar_set_running(True)
        # run_pipeline marks profiling as "running" after reset() — fire and forget.
        asyncio.create_task(run_pipeline(config))
        await asyncio.sleep(0)   # let sync preamble execute before refreshing
        flow_diagram.update(state.step_status, state.branch_taken)
        status_feed.render.refresh()

    def handle_stop():
        pass  # state.stop already set by the Stop button in sidebar

    def handle_clear():
        _sidebar_set_running(False)
        flow_diagram.update(state.step_status, state.branch_taken)
        status_feed.render.refresh()

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.header().classes("items-center justify-between px-6 py-3").style(
        header_style()
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("auto_awesome").classes("text-amber-400 text-2xl")
            ui.label("Agentic ML System").classes(
                "text-xl font-bold text-white tracking-tight"
            )
        ui.label("v1.0").classes("text-slate-500 text-sm font-mono")

    # ── Body: sidebar + main content side-by-side ────────────────────────────
    # Plain HTML row — avoids ui.left_drawer which polls JS state every tick
    # and floods the console with "JavaScript did not respond within 1.0 s".
    with ui.row().classes("w-full flex-nowrap").style("min-height: calc(100vh - 56px);"):
        # Sidebar column — fixed width, scrollable
        with ui.column().classes("flex-none overflow-y-auto p-4").style(sidebar_style()):
            sidebar.render(
                on_run=handle_run,
                on_stop=handle_stop,
                on_clear=handle_clear,
            )

        # Main content column — fills remaining space
        with ui.column().classes("flex-1 gap-4 p-6 overflow-y-auto").style(page_style()):
            flow_diagram.render()
            status_feed.render()

    # ── Drain timer — pass async fn directly so NiceGUI keeps slot context ────
    # Do NOT wrap in asyncio.create_task() — that strips the context and breaks
    # ui.notify / ui.download.
    timer = ui.timer(0.1, _drain_queue)

    # ✅ Attach disconnect handler to current client
    client = ui.context.client

    def _cleanup():
        try:
            timer.cancel()
        except Exception:
            pass

    client.on_disconnect(_cleanup)
