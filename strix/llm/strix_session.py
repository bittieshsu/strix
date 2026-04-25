"""``StrixSession`` — Session wrapper that runs the MemoryCompressor.

Delegates storage to any underlying session implementation (in-memory,
SQLite, Redis, …) and intercepts ``get_items`` so the
``MemoryCompressor`` runs before the model sees the history.

Wrapping (rather than reimplementing) keeps the pentest-tuned
summarization prompt and 90K-token budget intact. ``get_items`` is
also the last hook before ``call_model_input_filter``, so compressing
here means the filter sees a compressed history too.

If compression raises, we fall back to the uncompressed history and
flip a flag so future attempts skip — a permanently broken compressor
mustn't infinite-loop the agent into context starvation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from agents.memory.session import SessionABC


if TYPE_CHECKING:
    from agents.items import TResponseInputItem

    from strix.llm.memory_compressor import MemoryCompressor


logger = logging.getLogger(__name__)


class StrixSession(SessionABC):
    """Wraps an underlying ``SessionABC`` with Strix's memory compressor.

    The wrapped session owns persistence; ``StrixSession`` only intercepts
    ``get_items`` to run compression. Writes (``add_items``, ``pop_item``,
    ``clear_session``) pass through verbatim.

    On compressor failure, the call returns the uncompressed history and
    a per-instance flag is set so subsequent ``get_items`` calls skip the
    compressor entirely. This avoids an infinite "compress → fail → grow"
    loop when the compressor LLM is itself unavailable.
    """

    def __init__(
        self,
        underlying: SessionABC,
        compressor: MemoryCompressor,
    ) -> None:
        self._underlying = underlying
        self._compressor = compressor
        self._compression_disabled = False
        # ``SessionABC.session_id`` is a plain ``str`` field; pass through.
        self.session_id: str = getattr(underlying, "session_id", "strix-session")
        self.session_settings = getattr(underlying, "session_settings", None)

    @property
    def compression_disabled(self) -> bool:
        """True after the compressor has failed at least once on this session."""
        return self._compression_disabled

    async def get_items(
        self,
        limit: int | None = None,
    ) -> list[TResponseInputItem]:
        """Read items from underlying storage and (optionally) compress.

        On any compressor exception, log and return the uncompressed list.
        Set ``_compression_disabled`` so the next call short-circuits.
        """
        items = await self._underlying.get_items(limit=limit)
        if self._compression_disabled or not items:
            return items
        try:
            # Compressor expects ``list[dict[str, Any]]``; SDK's
            # ``TResponseInputItem`` is a TypedDict union — structurally
            # compatible. Compressor mutates content but preserves shape.
            compressed = self._compressor.compress_history(
                cast("list[dict[str, Any]]", items),
            )
            return cast("list[TResponseInputItem]", compressed)
        except Exception:
            logger.exception(
                "MemoryCompressor failed; returning uncompressed history. "
                "Compression disabled for this session for the rest of the run.",
            )
            self._compression_disabled = True
            return items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self._underlying.add_items(items)

    async def pop_item(self) -> TResponseInputItem | None:
        return await self._underlying.pop_item()

    async def clear_session(self) -> None:
        await self._underlying.clear_session()
