"""C6 regression test — concurrent notes JSONL writes must produce valid JSONL.

This test would fail before the C6 fix (AUDIT_R2 §1.1, applied in Phase 2.2):
the legacy ``_append_note_event`` opened the file and called ``f.write``
without holding ``_notes_lock``, so two threads writing simultaneously
could interleave bytes mid-line and corrupt the JSONL.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from strix.tools.notes.notes_actions import _append_note_event


@pytest.fixture
def notes_path(tmp_path: Path) -> Iterator[Path]:
    """Point ``_get_notes_jsonl_path`` at a tmp file for the test."""
    target = tmp_path / "notes" / "notes.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)

    with patch(
        "strix.tools.notes.notes_actions._get_notes_jsonl_path",
        return_value=target,
    ):
        yield target


def test_concurrent_note_writes_yield_valid_jsonl(notes_path: Path) -> None:
    """C6 fix: 50 threads x 20 events = 1000 lines, all valid JSON.

    Without the lock, byte-level interleaving on the file produces
    fragments like ``{"timesta{"timestamp"...`` that fail json.loads.
    """

    def writer(thread_idx: int) -> None:
        for i in range(20):
            note: dict[str, Any] = {
                "title": f"thread-{thread_idx}-note-{i}",
                "content": "x" * 200,  # non-trivial body to widen the race
                "category": "general",
            }
            _append_note_event(
                op="create",
                note_id=f"t{thread_idx}-i{i}",
                note=note,
            )

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = notes_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1000, f"expected 1000 lines, got {len(lines)}"
    for line in lines:
        # raises if the line is malformed JSON
        event = json.loads(line)
        assert event["op"] == "create"
        assert "note_id" in event


def test_single_writer_still_works(notes_path: Path) -> None:
    """Sanity: serial writes still produce a valid JSONL log."""
    _append_note_event("create", "n1", {"title": "first"})
    _append_note_event("update", "n1", {"title": "first updated"})
    _append_note_event("delete", "n1")

    events = [json.loads(line) for line in notes_path.read_text().splitlines()]
    assert [e["op"] for e in events] == ["create", "update", "delete"]
    assert all(e["note_id"] == "n1" for e in events)
