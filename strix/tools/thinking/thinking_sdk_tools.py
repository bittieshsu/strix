"""SDK function-tool wrapper for the legacy ``think`` tool.

Pattern: thin async wrapper that delegates to the legacy implementation
in :mod:`strix.tools.thinking.thinking_actions`. The legacy function is
sync and pure (no I/O), so we don't even need ``asyncio.to_thread``.

Validates the simplest tool-port pattern: legacy function in, JSON string
out, no sandbox involvement.
"""

from __future__ import annotations

import json

from strix.tools._decorator import strix_tool
from strix.tools.thinking.thinking_actions import think as _legacy_think


@strix_tool(timeout=10)
async def think(thought: str) -> str:
    """Record a private chain-of-thought note without taking any action.

    The "think" tool is the planning escape hatch for situations where a
    message-without-tool-call would otherwise halt the run (per the
    interactive-mode tool-call requirement). The thought itself is
    recorded but produces no side effects.

    Args:
        thought: The agent's reasoning to record. Must be non-empty.
    """
    result = _legacy_think(thought)
    return json.dumps(result, ensure_ascii=False)
