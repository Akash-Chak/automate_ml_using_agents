# nicegui_app/mlflow_server.py
#
# Singleton for the MLflow UI subprocess.
# Both pipeline_runner.py (auto-start on run) and sidebar.py (button click)
# go through this module so only one process is ever spawned.

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
from typing import Optional

_proc: Optional[subprocess.Popen] = None


def is_port_open(port: int = 5000) -> bool:
    """Return True if something is already listening on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def is_starting() -> bool:
    """Return True if our process was launched and is still alive (not yet listening)."""
    return _proc is not None and _proc.poll() is None


def ensure_started(project_root: str) -> bool:
    """
    Start `mlflow ui` if not already running or starting.

    Returns:
        True  — server is already listening on port 5000 (safe to open tab)
        False — process was just launched (or is still starting); caller should
                wait a moment before opening the tab
    """
    global _proc

    if is_port_open():
        return True   # already up

    if is_starting():
        print("[MLflow] Process is already starting (pid %d)…" % _proc.pid)
        return False  # don't spawn a second one

    # Convert Windows path to file:/// URI so MLflow's URI parser doesn't
    # mistake the drive letter (e.g. "C") for an unknown URI scheme.
    mlruns_path = pathlib.Path(project_root, "mlruns").as_uri()
    log_path = os.path.join(project_root, "mlflow_server.log")
    log_file = open(log_path, "w")
    _proc = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui",
         "--backend-store-uri", mlruns_path,
         "--host", "127.0.0.1",
         "--port", "5000",
         "--workers", "1"],
        stdout=log_file,
        stderr=log_file,
    )
    print(f"[MLflow] Starting server (pid {_proc.pid}) — logs: {log_path}")
    return False
