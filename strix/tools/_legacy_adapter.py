"""Shim that lets SDK function tools call legacy ``agent_state``-style functions.

The legacy harness's tools (notes, todos, reporting, …) take an
``agent_state`` argument with shape ``state.agent_id`` for per-agent silo
keying. Under the SDK migration the equivalent identity lives in
``RunContextWrapper.context["agent_id"]``.

Rather than rewrite every tool body, SDK function-tool wrappers build a
tiny adapter from the context dict and pass it to the legacy function.
The legacy code path remains untouched (the legacy executor still calls
its tools with the real ``AgentState``).

Used by:
    - ``tools/todo/todo_sdk_tools.py``
    - ``tools/notes/notes_sdk_tools.py``
    - ``tools/reporting/reporting_sdk_tools.py``
    - any other local tool that closes over ``agent_state.agent_id``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agents import RunContextWrapper


@dataclass
class LegacyAgentStateAdapter:
    """Just enough surface for legacy tools that read ``state.agent_id``.

    Don't rely on this for new code — it's only here to avoid touching
    the legacy ``*_actions.py`` modules during the migration. New SDK
    tools should read ``ctx.context["agent_id"]`` directly.
    """

    agent_id: str


def adapter_from_ctx(
    ctx: RunContextWrapper,
    default_agent_id: str = "sdk-default",
) -> LegacyAgentStateAdapter:
    """Build a ``LegacyAgentStateAdapter`` from an SDK run context.

    Falls back to ``default_agent_id`` when context is missing or its
    ``agent_id`` is unset — keeps tests and CLI dry-runs working without
    a fully-populated context.
    """
    inner = getattr(ctx, "context", None)
    if isinstance(inner, dict):
        agent_id = inner.get("agent_id") or default_agent_id
    else:
        agent_id = default_agent_id
    return LegacyAgentStateAdapter(agent_id=str(agent_id))
