"""``post_to_sandbox`` — host-to-container HTTP transport for sandbox tools.

Every Strix tool that runs inside the Kali container (browser,
terminal, python, file_edit, the seven Caido tools) has the same wire
shape: POST JSON to ``http://localhost:{tool_server_host_port}/execute``
with a Bearer token and ``{"agent_id", "tool_name", "kwargs"}`` body.

The helper centralizes timeouts (``connect=10s`` / ``read=150s``), a
50 MB response-size cap so a runaway tool can't OOM the host, and
predictable error-string shaping so transport failures don't tear
down the run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from agents import RunContextWrapper


logger = logging.getLogger(__name__)


_SANDBOX_TIMEOUT = httpx.Timeout(connect=10.0, read=150.0, write=150.0, pool=150.0)

# Cap so a runaway tool body never blows up the host heap.
_MAX_RESPONSE_BYTES = 50 * 1024 * 1024  # 50 MB


def _ctx_dict(ctx: RunContextWrapper) -> dict[str, Any] | None:
    """Return ``ctx.context`` if it's a dict, else ``None``.

    Strix's runtime always passes a dict (``make_agent_context``); other
    callers might not. Be defensive so a sandbox tool never raises just
    because the context shape is wrong.
    """
    inner = getattr(ctx, "context", None)
    return inner if isinstance(inner, dict) else None


async def post_to_sandbox(
    ctx: RunContextWrapper,
    tool_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """POST a tool invocation to the in-container FastAPI tool server.

    Returns:
        On success: ``{"result": <whatever the tool returned>}``.
        On any failure: ``{"error": "<human-readable error string>"}``.

    Never raises. Tool authors call this and pass the return value
    straight to the model (or extract ``result`` for further shaping).
    """
    inner = _ctx_dict(ctx)
    if inner is None:
        return {"error": "Sandbox not initialized: context is missing or not a dict."}

    port = inner.get("tool_server_host_port")
    token = inner.get("sandbox_token")
    agent_id = inner.get("agent_id", "unknown")

    if not port or not token:
        return {"error": "Sandbox not initialized: tool server port or token missing."}

    url = f"http://127.0.0.1:{port}/execute"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {"agent_id": agent_id, "tool_name": tool_name, "kwargs": kwargs}

    try:
        async with httpx.AsyncClient(timeout=_SANDBOX_TIMEOUT) as client:
            response = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        return {
            "error": (f"Sandbox tool '{tool_name}' timed out after {_SANDBOX_TIMEOUT.read}s."),
        }
    except httpx.RequestError as e:
        # ConnectError, ReadError, NetworkError, etc.
        return {"error": f"Sandbox connection failed: {e!s}"[:300]}

    if response.status_code == 401:
        return {"error": "Sandbox authorization failed (Bearer token invalid)."}
    if response.status_code >= 400:
        return {
            "error": (
                f"Sandbox tool '{tool_name}' failed with HTTP "
                f"{response.status_code}: {response.text[:300]}"
            ),
        }

    # Cap response size before parsing so a 1 GB rogue payload never lands
    # in our heap. Most legitimate tool responses are well under 100 KB.
    raw = response.content
    if len(raw) > _MAX_RESPONSE_BYTES:
        return {
            "error": (f"Sandbox response too large ({len(raw)} bytes; max {_MAX_RESPONSE_BYTES})."),
        }

    try:
        data: Any = response.json()
    except ValueError:
        return {
            "error": (f"Sandbox tool '{tool_name}' returned non-JSON: {response.text[:200]}"),
        }

    if not isinstance(data, dict):
        return {"error": f"Sandbox tool '{tool_name}' returned non-object JSON."}

    return data
