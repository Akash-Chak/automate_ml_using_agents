# nicegui_app/pipeline_runner.py
#
# Runs graph.stream() in a ThreadPoolExecutor so it never blocks the
# NiceGUI asyncio event loop.  Events are posted onto an asyncio.Queue;
# the UI drain loop (ui.timer) picks them up and refreshes components.

from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import traceback

# Force matplotlib to use the non-interactive Agg backend before any agent
# imports it.  Without this, matplotlib defaults to the tkinter backend which
# is not thread-safe and crashes when called from the ThreadPoolExecutor.
import matplotlib
matplotlib.use("Agg")

# Make the project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.langgraph_pipeline import build_graph
from nicegui_app.app_state import pipeline_state, STEPS
from nicegui_app.mlflow_server import ensure_started as _mlflow_ensure_started

_executor = ThreadPoolExecutor(max_workers=1)


def _make_progress_callback(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Returns a synchronous callback that safely posts tuning events to the queue."""
    def callback(event: dict):
        asyncio.run_coroutine_threadsafe(
            queue.put(("tuning_event", event)), loop
        )
    return callback


async def run_pipeline(config: dict):
    """
    Main entry point called by the Run button handler.
    Builds the graph, streams it in a thread, posts events to the queue.
    """
    state = pipeline_state
    state.reset()
    state.running = True
    state.config  = config

    # Start MLflow UI in the background so it's ready by the time the pipeline
    # finishes and the user clicks "Open MLflow UI".
    _mlflow_ensure_started(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Mark the first node as running immediately so the diagram lights up
    import time as _time
    state.step_status["profiling"] = "running"
    state.step_start_times["profiling"] = _time.time()

    loop  = asyncio.get_event_loop()
    queue = state.event_queue

    graph = build_graph()

    initial_state = {
        **config,
        "_progress_callback": _make_progress_callback(queue, loop),
    }

    def _blocking_stream():
        try:
            for chunk in graph.stream(initial_state):
                if state.stop:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("pipeline_stopped", None)), loop
                    )
                    return

                node_name   = list(chunk.keys())[0]
                node_output = chunk[node_name]

                asyncio.run_coroutine_threadsafe(
                    queue.put(("node_complete", node_name, node_output)), loop
                )

            asyncio.run_coroutine_threadsafe(
                queue.put(("pipeline_done", None)), loop
            )

        except Exception as exc:
            error_msg = f"{str(exc)}\n\nTRACEBACK:\n{traceback.format_exc()}"

            asyncio.run_coroutine_threadsafe(
                queue.put(("pipeline_error", error_msg)), loop
            )

    await loop.run_in_executor(_executor, _blocking_stream)
