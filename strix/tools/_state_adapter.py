"""Adapter exposing ``ctx.context['agent_id']`` as ``state.agent_id``.

Several tool implementations still take an ``agent_state`` argument
that they read ``.agent_id`` off of for per-agent silo keying. The SDK
keeps that same identity in ``ctx.context['agent_id']``. Rather than
plumb a different parameter through every tool body, we build a tiny
adapter object from the run context.

Used by:
    - ``tools/todo/tools.py``
    - ``tools/load_skill/tool.py``
    - ``tools/finish/tool.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agents import RunContextWrapper


@dataclass
class AgentStateAdapter:
    """Just enough surface for tools that read ``state.agent_id``."""

    agent_id: str


def adapter_from_ctx(
    ctx: RunContextWrapper,
    default_agent_id: str = "default",
) -> AgentStateAdapter:
    """Build an ``AgentStateAdapter`` from an SDK run context.

    Falls back to ``default_agent_id`` when context is missing or its
    ``agent_id`` is unset — keeps tests and CLI dry-runs working without
    a fully-populated context.
    """
    inner = getattr(ctx, "context", None)
    if isinstance(inner, dict):
        agent_id = inner.get("agent_id") or default_agent_id
    else:
        agent_id = default_agent_id
    return AgentStateAdapter(agent_id=str(agent_id))
