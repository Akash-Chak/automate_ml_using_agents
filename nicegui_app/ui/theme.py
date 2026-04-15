# nicegui_app/ui/theme.py
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    VISUAL CONFIGURATION — EDIT HERE                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# All colours, sizes, and style strings for the Agentic ML System UI live
# here.  Nothing in layout.py / flow_diagram.py / status_feed.py / sidebar.py
# needs to change for a visual tweak — modify this file only.
#
# Sections
#   1. Page & chrome colours
#   2. Node state colours  (flow diagram)
#   3. Status-feed icon colours
#   4. Flow diagram sizing  (node width, chart height, spacing)
#   5. Inline CSS helpers   (used with .style(...) in NiceGUI)
#   6. Tailwind class helpers (used with .classes(...) in NiceGUI)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PAGE & CHROME COLOURS
# ─────────────────────────────────────────────────────────────────────────────

PAGE_BG          = "#060b14"   # outermost page background
CARD_BG          = "#0d1117"   # cards (pipeline diagram, status feed)
CARD_BORDER      = "#1e2535"   # card / drawer border
HEADER_BG        = "#0a0f1a"   # top header bar
HEADER_BORDER    = "#1e2535"

SIDEBAR_WIDTH    = "260px"     # left drawer width
SIDEBAR_PADDING  = "20px"

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NODE STATE COLOURS  (flow diagram ECharts graph)
#
#     pending  — node not yet reached
#     running  — currently executing  (amber pulse + glow)
#     done     — finished successfully (green)
#     skipped  — branch not taken      (dimmed grey)
#     error    — agent threw an exception (red)
# ─────────────────────────────────────────────────────────────────────────────

NODE_FILL = {
    "pending": "#1e2d3d",
    "running": "#f59e0b",   # amber
    "done":    "#22c55e",   # green
    "skipped": "#374151",   # dark grey
    "error":   "#ef4444",   # red
}

# Text colour *inside* the node box
NODE_TEXT = {
    "pending": "#64748b",   # muted slate
    "running": "#1c1917",   # near-black (dark text on amber)
    "done":    "#14532d",   # dark green
    "skipped": "#4b5563",   # grey
    "error":   "#7f1d1d",   # dark red
}

# Glow colour shown as ECharts shadowColor when a node is running
NODE_GLOW_COLOR   = "#f59e0b"   # amber glow
NODE_GLOW_BLUR    = 20          # shadowBlur radius (px)

# Edge colours
EDGE_DONE         = "#22c55e"   # solid green — source node completed
EDGE_PENDING      = "#334155"   # solid grey  — not yet reached
EDGE_BRANCH_TAKEN = "#22c55e"   # solid green — this branch was selected
EDGE_BRANCH_SKIP  = "#1f2937"   # near-black dashed — other branch

# Legend entries  (label → fill colour)
LEGEND_ITEMS = [
    ("Pending", NODE_FILL["pending"]),
    ("Running", NODE_FILL["running"]),
    ("Done",    NODE_FILL["done"]),
    ("Skipped", NODE_FILL["skipped"]),
    ("Error",   NODE_FILL["error"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  STATUS-FEED ICON COLOURS  (Tailwind text- classes)
# ─────────────────────────────────────────────────────────────────────────────

ICON_COLOR_DEFAULT  = "text-green-400"    # completed step
ICON_COLOR_ERROR    = "text-red-400"      # error / error icon
ICON_COLOR_RUNNING  = "text-amber-400"    # timer / settings / pending
ICON_COLOR_UPDATE   = "text-blue-400"     # trending_up (best score update)
ICON_COLOR_DECISION = "text-purple-400"   # fork_right (decision nodes)

# Map icon name → colour class
ICON_COLOR_MAP: dict[str, str] = {
    "error":       ICON_COLOR_ERROR,
    "settings":    ICON_COLOR_RUNNING,
    "timer":       ICON_COLOR_RUNNING,
    "pending":     ICON_COLOR_RUNNING,   # heartbeat tick
    "trending_up": ICON_COLOR_UPDATE,
    "fork_right":  ICON_COLOR_DECISION,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FLOW DIAGRAM SIZING
# ─────────────────────────────────────────────────────────────────────────────

DIAGRAM_NODE_SIZE    = [110, 42]    # [width_px, height_px] — ECharts symbolSize
DIAGRAM_FONT_SIZE    = 11           # label font size inside nodes
DIAGRAM_CHART_HEIGHT = "300px"      # CSS height of the ECharts canvas
DIAGRAM_CURVENESS    = 0.2          # edge curve amount (0 = straight)
DIAGRAM_EDGE_ARROW   = 8            # arrowhead size (px)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  INLINE CSS HELPERS  — pass these to .style("...")
# ─────────────────────────────────────────────────────────────────────────────

def card_style() -> str:
    return (
        f"background:{CARD_BG}; "
        f"border:1px solid {CARD_BORDER}; "
        "border-radius:12px; padding:16px;"
    )

def header_style() -> str:
    return (
        f"background:{HEADER_BG}; "
        f"border-bottom:1px solid {HEADER_BORDER};"
    )

def sidebar_style() -> str:
    return (
        f"background:{CARD_BG}; "
        f"border-right:1px solid {CARD_BORDER}; "
        f"padding:{SIDEBAR_PADDING}; "
        f"width:{SIDEBAR_WIDTH};"
    )

def page_style() -> str:
    return f"background:{PAGE_BG}; min-height:100vh;"

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TAILWIND CLASS HELPERS  — pass these to .classes("...")
# ─────────────────────────────────────────────────────────────────────────────

LABEL_SECTION      = "font-bold text-slate-300 text-sm"
LABEL_HINT         = "text-xs text-slate-500"
LABEL_STEP_HEADER  = "text-xs text-slate-500 font-mono uppercase tracking-wide"
LABEL_STEP_TEXT    = "text-sm text-slate-200 font-mono"
LABEL_CARD_TITLE   = "text-lg font-bold text-slate-300"
LABEL_ERROR        = "text-red-400 text-sm font-mono"
LABEL_MUTED        = "text-slate-500 text-sm font-mono"
