"""Visualization layer — Context MRI and friends.

Optional extra: ``pip install "ctxbudgeter[viz]"``.

All renderers produce **self-contained, local HTML** with inline CSS/SVG and
**no remote JavaScript**. Secrets are never embedded — only masked scanner
previews flow through. The "Influence Proxy" shown in Context MRI is NOT model
attention; it is a transparent score computed from metadata (see the disclaimer
rendered in every report).

The core renderers depend only on the standard library, so they work even without
the optional ``[viz]`` dependencies installed; the extra pins plotly/jinja2/
networkx/rich for users who want to extend the visualizations.
"""

from __future__ import annotations

from .diff_viz import ContextDiffViz
from .mcp_viz import MCPToolViz
from .mri import ContextMRI

__all__ = ["ContextDiffViz", "ContextMRI", "MCPToolViz"]
