"""SDK function-tool wrappers for the legacy todo tools.

Six tools, all in-memory, all per-agent (keyed by ``ctx.context["agent_id"]``
through :class:`LegacyAgentStateAdapter`). Bulk forms are preserved —
``todos`` / ``updates`` / ``todo_ids`` accept JSON strings or comma-separated
strings the same way the legacy XML schema documented.

Pattern: thin async wrappers that delegate to the legacy implementations
in :mod:`strix.tools.todo.todo_actions`. Legacy code is untouched.
"""

from __future__ import annotations

import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._state_adapter import adapter_from_ctx
from strix.tools.todo import todo_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    """JSON-dump a legacy result dict for the model. ``ensure_ascii=False``
    so unicode flows through; ``default=str`` to handle stray datetimes."""
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=30)
async def create_todo(
    ctx: RunContextWrapper,
    title: str | None = None,
    description: str | None = None,
    priority: str = "normal",
    todos: str | None = None,
) -> str:
    """Create one or many todos for the current agent.

    Args:
        title: Title of a single todo (alternative to bulk ``todos``).
        description: Optional details for the single todo.
        priority: ``"low" | "normal" | "high" | "critical"``.
        todos: Optional JSON string or comma-separated list for bulk create.
    """
    state = adapter_from_ctx(ctx)
    return _dump(
        _impl.create_todo(
            agent_state=state,
            title=title,
            description=description,
            priority=priority,
            todos=todos,
        ),
    )


@strix_tool(timeout=30)
async def list_todos(
    ctx: RunContextWrapper,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    """List the current agent's todos, sorted by status then priority.

    Args:
        status: Optional ``"pending" | "in_progress" | "done"`` filter.
        priority: Optional ``"low" | "normal" | "high" | "critical"`` filter.
    """
    state = adapter_from_ctx(ctx)
    return _dump(_impl.list_todos(agent_state=state, status=status, priority=priority))


@strix_tool(timeout=30)
async def update_todo(
    ctx: RunContextWrapper,
    todo_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    updates: str | None = None,
) -> str:
    """Update one or many todos.

    Args:
        todo_id: Single-todo target (alternative to bulk ``updates``).
        title / description / priority / status: New values for the single
            todo. Omit to leave unchanged.
        updates: Bulk form — JSON list of update dicts.
    """
    state = adapter_from_ctx(ctx)
    return _dump(
        _impl.update_todo(
            agent_state=state,
            todo_id=todo_id,
            title=title,
            description=description,
            priority=priority,
            status=status,
            updates=updates,
        ),
    )


@strix_tool(timeout=30)
async def mark_todo_done(
    ctx: RunContextWrapper,
    todo_id: str | None = None,
    todo_ids: str | None = None,
) -> str:
    """Mark one (``todo_id``) or many (``todo_ids``) todos as done."""
    state = adapter_from_ctx(ctx)
    return _dump(
        _impl.mark_todo_done(agent_state=state, todo_id=todo_id, todo_ids=todo_ids),
    )


@strix_tool(timeout=30)
async def mark_todo_pending(
    ctx: RunContextWrapper,
    todo_id: str | None = None,
    todo_ids: str | None = None,
) -> str:
    """Mark one (``todo_id``) or many (``todo_ids``) todos as pending."""
    state = adapter_from_ctx(ctx)
    return _dump(
        _impl.mark_todo_pending(
            agent_state=state,
            todo_id=todo_id,
            todo_ids=todo_ids,
        ),
    )


@strix_tool(timeout=30)
async def delete_todo(
    ctx: RunContextWrapper,
    todo_id: str | None = None,
    todo_ids: str | None = None,
) -> str:
    """Delete one (``todo_id``) or many (``todo_ids``) todos."""
    state = adapter_from_ctx(ctx)
    return _dump(
        _impl.delete_todo(agent_state=state, todo_id=todo_id, todo_ids=todo_ids),
    )
