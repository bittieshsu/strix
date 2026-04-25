"""SDK function-tool wrapper for the legacy ``web_search`` tool.

The legacy ``web_search_actions.web_search`` is a synchronous Perplexity
API call (300s timeout, ``requests``). We wrap it with
``asyncio.to_thread`` so the call doesn't block the SDK event loop while
the API responds — same parity for the model, no surprises.

Pattern matches notes/todo/think wrappers from Phase 2.3.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools.web_search import web_search_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


# Perplexity request timeout in the legacy code is 300s; give the SDK
# tool a slightly larger budget so the network round-trip + JSON decode
# doesn't push us over the edge under load.
@strix_tool(timeout=330)
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web with Perplexity, scoped to security-relevant content.

    Returns a JSON-encoded ``{"success": bool, "content": str, ...}``
    dict matching the legacy shape exactly.

    Args:
        query: The search query. The legacy tool prepends a security-focused
            system prompt to bias results toward CVEs, exploits, and Kali-
            compatible commands.
    """
    return _dump(await asyncio.to_thread(_impl.web_search, query=query))
