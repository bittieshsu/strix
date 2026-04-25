"""Strix agent package.

Public surface:

- :func:`build_strix_agent` — assemble a root or child ``agents.Agent``.
- :func:`make_child_factory` — closure factory passed via context to
  the multi-agent ``create_agent`` graph tool.
- :func:`render_system_prompt` — render the Jinja system prompt.
"""

from .factory import build_strix_agent, make_child_factory
from .prompt import render_system_prompt


__all__ = [
    "build_strix_agent",
    "make_child_factory",
    "render_system_prompt",
]
