# nicegui_app/main.py
#
# Entry point.  Run with:
#   python nicegui_app/main.py
# or from the project root:
#   python -m nicegui_app.main

from __future__ import annotations

import os
import sys

# Make the project root importable regardless of where the script is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present (ANTHROPIC_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from nicegui import app, ui
from nicegui_app.ui.layout import build_page

# Serve static assets (styles.css, any future assets)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.add_static_files("/static", _static_dir)


@ui.page("/")
def index():
    # Load page-level CSS — edit nicegui_app/static/styles.css to change styles
    ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')
    build_page()


if __name__ in ("__main__", "__mp_main__"):
    ui.run(
        title="Agentic ML System",
        port=8080,
        dark=True,
        reload=False,
        favicon="🤖",
    )
