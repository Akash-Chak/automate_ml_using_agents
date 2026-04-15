# nicegui_app/ui/sidebar.py

from __future__ import annotations

import asyncio
import io
import os
import tempfile

import pandas as pd
from nicegui import ui

from nicegui_app.app_state import pipeline_state
from nicegui_app.mlflow_server import ensure_started as _mlflow_ensure_started
from nicegui_app.ui.theme import LABEL_SECTION, LABEL_HINT


def _detect_columns(path_or_bytes, is_bytes: bool = False) -> list[str]:
    try:
        if is_bytes:
            df = pd.read_csv(io.BytesIO(path_or_bytes), nrows=5)
        else:
            df = pd.read_csv(path_or_bytes, nrows=5)
        return list(df.columns)
    except Exception:
        return []


def _auto_detect_problem_type(path_or_bytes, target: str, is_bytes: bool = False) -> str:
    try:
        if is_bytes:
            df = pd.read_csv(io.BytesIO(path_or_bytes), usecols=[target])
        else:
            df = pd.read_csv(path_or_bytes, usecols=[target])
        col = df[target]
        if col.dtype == object or col.nunique() < 10:
            return "classification"
        return "regression"
    except Exception:
        return "classification"


def _notify_safe(message: str, **kwargs) -> None:
    """Call ui.notify, silently swallowing NiceGUI context errors.

    Context can become stale after long async waits (e.g. the mlflow poll loop).
    Falling back to a console print keeps the error visible without crashing.
    """
    try:
        ui.notify(message, **kwargs)
    except RuntimeError:
        print(f"[MLflow] {message}")


async def _open_mlflow():
    """Open MLflow UI in a new tab.

    Delegates process management to mlflow_server.py (singleton — no double-start).
    Polls localhost:5000 until the server is ready (up to 30 s) before opening
    the tab, so we never redirect to a dead server.
    """
    import socket as _socket

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    already_up = _mlflow_ensure_started(project_root)

    if not already_up:
        _notify_safe("MLflow starting — waiting for server…", color="info", timeout=35000)

        # Poll every 500 ms, up to 30 s (60 attempts)
        ready = False
        for _ in range(60):
            await asyncio.sleep(0.5)
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", 5000)) == 0:
                    ready = True
                    break

        if not ready:
            _notify_safe(
                "MLflow did not start within 30 s. Check mlflow_server.log in the project root.",
                color="negative",
                timeout=0,
            )
            return

    try:
        ui.navigate.to("http://localhost:5000", new_tab=True)
    except RuntimeError:
        print("[MLflow] Open http://localhost:5000 in your browser")


# Module-level references to disableable elements — populated by render()
# and used by set_running() which is called from the drain loop in layout.py.
_lockable: list = []


def set_running(running: bool) -> None:
    """
    Called by the drain loop whenever the pipeline starts, completes,
    stops, or errors.  Disables all config inputs + the Run button while
    running so the user can't accidentally launch a second pipeline.
    """
    for el in _lockable:
        if running:
            el.disable()
        else:
            el.enable()


def render(
    on_run: callable,
    on_stop: callable,
    on_clear: callable,
):
    global _lockable
    state = pipeline_state
    _lockable = []          # reset on each render (page reload / clear)

    # ── CSV source ────────────────────────────────────────────────────────────
    ui.label("Data Source").classes(f"{LABEL_SECTION} mt-2")

    csv_path_input = ui.input(
        "Local CSV path",
        placeholder="data.csv",
    ).classes("w-full").style("font-size:13px;")
    _lockable.append(csv_path_input)

    # Upload alternative
    upload_bytes: dict = {}  # mutable container for uploaded content

    async def handle_upload(e):
        # NiceGUI 3.x: e.file is a FileUpload object; read() is async
        content = await e.file.read()
        name    = e.file.name
        upload_bytes["content"] = content
        upload_bytes["name"]    = name
        cols = _detect_columns(content, is_bytes=True)
        state.detected_columns = cols
        target_select.options = cols
        target_select.value   = cols[0] if cols else None
        ui.notify(f"Loaded {name}: {len(cols)} columns", color="positive")

    ui.upload(
        label="or upload CSV",
        on_upload=handle_upload,
        auto_upload=True,
    ).classes("w-full mt-1").props("accept=.csv flat dense")

    ui.separator().classes("my-3")

    # ── Target + problem type ─────────────────────────────────────────────────
    ui.label("Target Column").classes(LABEL_SECTION)

    target_select = ui.select(
        options=state.detected_columns or [],
        label="Select target column",
    ).classes("w-full").style("font-size:13px;")
    _lockable.append(target_select)

    # Allow typing a column name manually if not yet loaded
    target_input = ui.input(
        "or type column name",
        placeholder="target",
    ).classes("w-full mt-1").style("font-size:13px;")
    _lockable.append(target_input)

    ui.label("Problem Type").classes("font-bold text-slate-300 text-sm mt-3")
    problem_toggle = ui.toggle(
        ["classification", "regression"],
        value="classification",
    ).classes("w-full")
    _lockable.append(problem_toggle)

    ui.separator().classes("my-3")

    # ── Tuning mode ───────────────────────────────────────────────────────────
    ui.label("Tuning Mode").classes(LABEL_SECTION)
    tuning_select = ui.select(
        options=["smoke_test", "full_search"],
        value="smoke_test",
        label="Mode",
    ).classes("w-full").style("font-size:13px;")
    _lockable.append(tuning_select)

    ui.label(
        "smoke_test: 1 trial/model (fast)\nfull_search: Optuna TPE"
    ).classes(f"{LABEL_HINT} mt-1 whitespace-pre-line")

    ui.separator().classes("my-3")

    # ── Load columns button (for local path) ─────────────────────────────────
    def handle_load_path():
        path = csv_path_input.value.strip()
        if not path:
            ui.notify("Enter a CSV path first.", color="warning")
            return
        if not os.path.exists(path):
            ui.notify(f"File not found: {path}", color="negative")
            return
        cols = _detect_columns(path)
        if not cols:
            ui.notify("Could not read columns.", color="negative")
            return
        state.detected_columns = cols
        target_select.options  = cols
        target_select.value    = cols[0] if cols else None
        ui.notify(f"Loaded {len(cols)} columns", color="positive")

    load_btn = ui.button("Load columns", on_click=handle_load_path, icon="refresh") \
      .classes("w-full").props("flat dense").style("font-size:12px;")
    _lockable.append(load_btn)

    ui.separator().classes("my-3")

    # ── Run / Stop / Clear ────────────────────────────────────────────────────
    def build_config() -> dict | None:
        path   = csv_path_input.value.strip()
        target = target_select.value or target_input.value.strip()

        if not target:
            ui.notify("Select or type a target column.", color="warning")
            return None

        if upload_bytes.get("content"):
            # Save uploaded file to a temp file so agents can read it
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", mode="wb"
            )
            tmp.write(upload_bytes["content"])
            tmp.flush()
            path = tmp.name
        elif not path:
            ui.notify("Provide a CSV path or upload a file.", color="warning")
            return None
        elif not os.path.exists(path):
            ui.notify(f"File not found: {path}", color="negative")
            return None

        return {
            "dataset_path":  path,
            "target_column": target,
            "problem_type":  problem_toggle.value,
            "tuning_mode":   tuning_select.value,
            "objective":     "Maximise predictive performance",
        }

    async def handle_run():
        if state.running:
            ui.notify("Pipeline already running.", color="warning")
            return
        config = build_config()
        if config is None:
            return
        await on_run(config)

    def handle_stop():
        state.stop = True
        ui.notify("Stop signal sent — pipeline will halt after the current agent.", color="info")
        on_stop()

    def handle_clear():
        state.reset()
        on_clear()
        ui.notify("Cleared.", color="info")

    with ui.row().classes("w-full gap-2"):
        run_btn = ui.button("Run", on_click=handle_run, icon="play_arrow") \
          .classes("flex-1 bg-blue-700 text-white").style("border-radius:8px;")
        _lockable.append(run_btn)
        ui.button("Stop", on_click=handle_stop, icon="stop") \
          .classes("flex-1").props("flat").style("border-radius:8px;")

    ui.button("Clear", on_click=handle_clear, icon="refresh") \
      .classes("w-full mt-1").props("flat dense").style("font-size:12px;")

    ui.separator().classes("my-4")

    # ── MLflow launch ─────────────────────────────────────────────────────────
    ui.button(
        "Open MLflow UI",
        on_click=_open_mlflow,
        icon="open_in_new",
    ).classes("w-full").props("outline").style(
        "border-radius:8px; border-color:#334155; font-size:12px;"
    )
    ui.label("Auto-starts mlflow ui if not running").classes(
        f"{LABEL_HINT} text-center mt-1"
    )
