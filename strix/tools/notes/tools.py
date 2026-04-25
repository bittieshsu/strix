"""SDK function-tool wrappers for the legacy notes tools.

Five tools, all module-global (no per-agent silo). The legacy
``notes_actions.py`` module already implements JSONL persistence and
wiki Markdown rendering; these wrappers are pure delegation.

The C6 fix (lock-protected JSONL writes) was applied directly to the
legacy module, so both code paths benefit.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools.notes import notes_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=30)
async def create_note(
    ctx: RunContextWrapper,
    title: str,
    content: str,
    category: str = "general",
    tags: list[str] | None = None,
) -> str:
    """Create a note in the current run's notes store.

    Notes are persisted to ``run_dir/notes/notes.jsonl`` and (for the
    ``wiki`` category) rendered as Markdown to ``run_dir/wiki/<slug>.md``.

    Args:
        title: Required, non-empty title.
        content: Note body. Markdown is preserved.
        category: One of ``"general" | "findings" | "methodology" |
            "questions" | "plan" | "wiki"``.
        tags: Optional list of free-form tags.
    """
    # The legacy function does file I/O under a threading.RLock.
    # Wrap in to_thread so we don't block the event loop while waiting
    # on the lock or fsync.
    result = await asyncio.to_thread(
        _impl.create_note,
        title=title,
        content=content,
        category=category,
        tags=tags,
    )
    return _dump(result)


@strix_tool(timeout=30)
async def list_notes(
    ctx: RunContextWrapper,
    category: str | None = None,
    tags: list[str] | None = None,
    search: str | None = None,
    include_content: bool = False,
) -> str:
    """List notes, optionally filtered.

    Args:
        category: Filter by category.
        tags: Filter to notes that have any of these tags.
        search: Substring match against title and content.
        include_content: When False (default), entries get a ``content_preview``;
            when True, full content is included.
    """
    result = await asyncio.to_thread(
        _impl.list_notes,
        category=category,
        tags=tags,
        search=search,
        include_content=include_content,
    )
    return _dump(result)


@strix_tool(timeout=30)
async def get_note(ctx: RunContextWrapper, note_id: str) -> str:
    """Fetch one note by its 5-char ID. Returns full content."""
    result = await asyncio.to_thread(_impl.get_note, note_id=note_id)
    return _dump(result)


@strix_tool(timeout=30)
async def update_note(
    ctx: RunContextWrapper,
    note_id: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update a note's title, content, or tags. Pass ``None`` to leave a field unchanged."""
    result = await asyncio.to_thread(
        _impl.update_note,
        note_id=note_id,
        title=title,
        content=content,
        tags=tags,
    )
    return _dump(result)


@strix_tool(timeout=30)
async def delete_note(ctx: RunContextWrapper, note_id: str) -> str:
    """Delete a note. For wiki notes, also removes the rendered Markdown file."""
    result = await asyncio.to_thread(_impl.delete_note, note_id=note_id)
    return _dump(result)
