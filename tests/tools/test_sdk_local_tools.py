"""Phase 2.3 smoke tests for the simplest SDK-wrapped local tools.

Validates the wrapping pattern (legacy implementation in, JSON string out)
on three tool families: think (trivial), todo (in-memory + agent_state
adapter), notes (in-memory + JSONL persistence).

If this slice works end-to-end the same pattern carries the rest of the
local tools (reporting, web_search, file_edit, finish_scan, load_skill).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from agents.tool import FunctionTool

from strix.tools.notes import notes_actions as _notes_legacy
from strix.tools.notes.notes_sdk_tools import (
    create_note,
    delete_note,
    get_note,
    list_notes,
    update_note,
)
from strix.tools.thinking.thinking_sdk_tools import think
from strix.tools.todo.todo_sdk_tools import (
    create_todo,
    delete_todo,
    list_todos,
    mark_todo_done,
    mark_todo_pending,
    update_todo,
)


@dataclass
class _Ctx:
    """Stand-in for ``RunContextWrapper``."""

    context: dict[str, Any] = field(default_factory=dict)


def _ctx_for(agent_id: str = "test-agent") -> _Ctx:
    return _Ctx(context={"agent_id": agent_id})


async def _invoke(tool: FunctionTool, ctx: _Ctx, **kwargs: Any) -> dict[str, Any]:
    """Invoke a function tool the way the SDK would and JSON-decode the result."""
    from agents.tool_context import ToolContext

    tool_ctx = ToolContext(
        context=ctx.context,
        usage=None,
        tool_name=tool.name,
        tool_call_id="test-call-id",
        tool_arguments=json.dumps(kwargs),
    )
    result = await tool.on_invoke_tool(tool_ctx, json.dumps(kwargs))
    assert isinstance(result, str)
    decoded = json.loads(result)
    assert isinstance(decoded, dict)
    return decoded


# --- think ----------------------------------------------------------------


def test_think_is_a_function_tool() -> None:
    assert isinstance(think, FunctionTool)
    assert think.name == "think"


@pytest.mark.asyncio
async def test_think_records_thought() -> None:
    ctx = _ctx_for()
    thought = "planning my next move"
    out = await _invoke(think, ctx, thought=thought)
    assert out["success"] is True
    assert f"{len(thought)} characters" in out["message"]


@pytest.mark.asyncio
async def test_think_rejects_empty() -> None:
    ctx = _ctx_for()
    out = await _invoke(think, ctx, thought="   ")
    assert out["success"] is False


# --- todo -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_todo_storage() -> None:
    """Each test starts with an empty todo store so tests don't bleed."""
    from strix.tools.todo import todo_actions

    todo_actions._todos_storage.clear()


def test_todo_tools_are_function_tools() -> None:
    for tool in (
        create_todo,
        list_todos,
        update_todo,
        mark_todo_done,
        mark_todo_pending,
        delete_todo,
    ):
        assert isinstance(tool, FunctionTool)


@pytest.mark.asyncio
async def test_todo_lifecycle() -> None:
    ctx = _ctx_for("agent-A")

    # Create
    created = await _invoke(create_todo, ctx, title="audit endpoint", priority="high")
    assert created["success"] is True
    assert created["count"] == 1
    todo_id = created["created"][0]["todo_id"]

    # List
    listed = await _invoke(list_todos, ctx)
    assert listed["success"] is True
    assert any(t["todo_id"] == todo_id for t in listed["todos"])

    # Update
    updated = await _invoke(update_todo, ctx, todo_id=todo_id, status="in_progress")
    assert updated["success"] is True

    # Mark done
    done = await _invoke(mark_todo_done, ctx, todo_id=todo_id)
    assert done["success"] is True

    # Reset to pending
    pending = await _invoke(mark_todo_pending, ctx, todo_id=todo_id)
    assert pending["success"] is True

    # Delete
    deleted = await _invoke(delete_todo, ctx, todo_id=todo_id)
    assert deleted["success"] is True


@pytest.mark.asyncio
async def test_todos_are_per_agent_isolated() -> None:
    """Two agents should have independent todo stores."""
    ctx_a = _ctx_for("agent-A")
    ctx_b = _ctx_for("agent-B")

    await _invoke(create_todo, ctx_a, title="A's task")
    await _invoke(create_todo, ctx_b, title="B's task")

    list_a = await _invoke(list_todos, ctx_a)
    list_b = await _invoke(list_todos, ctx_b)

    titles_a = [t["title"] for t in list_a["todos"]]
    titles_b = [t["title"] for t in list_b["todos"]]
    assert titles_a == ["A's task"]
    assert titles_b == ["B's task"]


@pytest.mark.asyncio
async def test_create_todo_bulk_via_json_string() -> None:
    ctx = _ctx_for()
    out = await _invoke(
        create_todo,
        ctx,
        todos=json.dumps(
            [
                {"title": "t1", "priority": "high"},
                {"title": "t2", "priority": "low"},
            ],
        ),
    )
    assert out["success"] is True
    assert out["count"] == 2


# --- notes ----------------------------------------------------------------


@pytest.fixture
def notes_run_dir(tmp_path: Path) -> Iterator[Path]:
    """Point the legacy notes module at a fresh run dir per test."""
    run_dir = tmp_path / "strix_runs" / "test"
    run_dir.mkdir(parents=True)
    _notes_legacy._notes_storage.clear()
    _notes_legacy._loaded_notes_run_dir = None

    with patch.object(_notes_legacy, "_get_run_dir", return_value=run_dir):
        yield run_dir


def test_notes_tools_are_function_tools() -> None:
    for tool in (create_note, list_notes, get_note, update_note, delete_note):
        assert isinstance(tool, FunctionTool)


@pytest.mark.asyncio
async def test_note_lifecycle(notes_run_dir: Path) -> None:
    ctx = _ctx_for()

    created = await _invoke(
        create_note,
        ctx,
        title="SQLi at /login",
        content="Form param `email` reflects into the WHERE clause.",
        category="findings",
        tags=["sqli", "auth"],
    )
    assert created["success"] is True
    note_id = created["note_id"]

    listed = await _invoke(list_notes, ctx, category="findings")
    assert listed["success"] is True
    assert listed["total_count"] == 1

    fetched = await _invoke(get_note, ctx, note_id=note_id)
    assert fetched["success"] is True
    assert "WHERE clause" in fetched["note"]["content"]

    updated = await _invoke(
        update_note,
        ctx,
        note_id=note_id,
        content="Confirmed boolean-blind SQLi.",
    )
    assert updated["success"] is True

    deleted = await _invoke(delete_note, ctx, note_id=note_id)
    assert deleted["success"] is True


@pytest.mark.asyncio
async def test_notes_jsonl_appended(notes_run_dir: Path) -> None:
    """Verify side effect: notes.jsonl receives one event per op."""
    ctx = _ctx_for()
    await _invoke(create_note, ctx, title="t", content="c", category="general")

    jsonl = notes_run_dir / "notes" / "notes.jsonl"
    assert jsonl.exists()
    events = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    assert events[0]["op"] == "create"
    assert events[0]["note"]["title"] == "t"
