"""StrixTracingProcessor — SDK trace processor that writes events.jsonl.

Hooks into the SDK's tracing pipeline and writes events to
``strix_runs/<run-name>/events.jsonl``. PII scrubbing via the existing
``TelemetrySanitizer``.

References:
    - PLAYBOOK.md §2.9
    - AUDIT_R2.md §1.2 (C7 — JSONL writes must be lock-protected)
    - AUDIT_R3.md C16 (writes must catch OSError; never tear down the run)
    - AUDIT_R3.md F3 (every TracingProcessor hook is SYNC, not async)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.tracing.processor_interface import TracingProcessor


if TYPE_CHECKING:
    from agents.tracing.spans import Span
    from agents.tracing.traces import Trace

    from strix.telemetry.utils import TelemetrySanitizer


logger = logging.getLogger(__name__)


# Module-level lock registry — one per JSONL file so two processors writing
# different run-dirs don't serialize unnecessarily, but two processors
# writing the *same* run-dir do.
_FILE_LOCKS: dict[Path, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _GUARD:
        return _FILE_LOCKS.setdefault(path, threading.Lock())


class StrixTracingProcessor(TracingProcessor):
    """Append trace + span events as JSONL into ``run_dir/events.jsonl``.

    Every hook is synchronous — required by ``TracingProcessor`` ABC.
    Every write is protected by a per-path ``threading.Lock`` so concurrent
    spans (e.g., from parallel agent tasks) cannot interleave bytes
    mid-line and corrupt the JSONL (C7).

    Every write is wrapped in ``try/except OSError`` so a full disk or a
    permission error during the run does NOT propagate up the hook chain
    and tear down the agent (C16).

    PII scrubbing via :class:`TelemetrySanitizer` runs on every event
    before it hits the file.
    """

    def __init__(
        self,
        run_dir: Path,
        sanitizer: TelemetrySanitizer | None = None,
    ) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir: Path = run_dir
        self.events_path: Path = run_dir / "events.jsonl"
        if sanitizer is None:
            from strix.telemetry.utils import TelemetrySanitizer

            sanitizer = TelemetrySanitizer()
        self.sanitizer: TelemetrySanitizer = sanitizer

    # --- Internal helpers -------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> None:
        """Sanitize ``event`` and append it as one JSONL line.

        Failures are swallowed — we'd rather lose a trace event than fail
        the run. Errors are logged at WARNING (C16).
        """
        try:
            clean = self.sanitizer.sanitize(event)
        except Exception:
            logger.exception("Trace event sanitization failed; dropping event")
            return
        try:
            with (
                _lock_for(self.events_path),
                self.events_path.open(
                    "a",
                    encoding="utf-8",
                ) as f,
            ):
                f.write(json.dumps(clean, ensure_ascii=True) + "\n")
        except OSError:
            logger.exception("Failed to append trace event to %s", self.events_path)

    @staticmethod
    def _span_kind(span: Span[Any]) -> str:
        """Map ``SomethingSpanData`` → ``"something"`` for the event_type."""
        name = type(span.span_data).__name__
        if name.endswith("SpanData"):
            name = name[: -len("SpanData")]
        return name.lower() or "span"

    @staticmethod
    def _trace_metadata(trace: Trace) -> dict[str, Any]:
        meta: dict[str, Any] = {"name": getattr(trace, "name", None)}
        # ``Trace.export()`` includes metadata + group_id when set.
        try:
            exported = trace.export()
            if isinstance(exported, dict):
                for key in ("metadata", "group_id", "workflow_name"):
                    if key in exported and exported[key] is not None:
                        meta[key] = exported[key]
        except Exception:
            logger.debug("trace.export failed", exc_info=True)
        return meta

    # --- TracingProcessor ABC --------------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        self._emit(
            {
                "event_type": "run.started",
                "trace_id": trace.trace_id,
                "metadata": self._trace_metadata(trace),
            }
        )

    def on_trace_end(self, trace: Trace) -> None:
        self._emit(
            {
                "event_type": "run.completed",
                "trace_id": trace.trace_id,
            }
        )

    def on_span_start(self, span: Span[Any]) -> None:
        kind = self._span_kind(span)
        self._emit(
            {
                "event_type": f"{kind}.started",
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "data": self._safe_export(span),
            }
        )

    def on_span_end(self, span: Span[Any]) -> None:
        kind = self._span_kind(span)
        self._emit(
            {
                "event_type": f"{kind}.completed",
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "data": self._safe_export(span),
            }
        )

    def force_flush(self) -> None:
        """All writes are synchronous; nothing to flush."""

    def shutdown(self) -> None:
        """No-op; nothing to release."""

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _safe_export(span: Span[Any]) -> dict[str, Any] | None:
        try:
            data = span.span_data.export()
            return data if isinstance(data, dict) else None
        except Exception:
            logger.debug("span_data.export failed for %s", span.span_id, exc_info=True)
            return None
