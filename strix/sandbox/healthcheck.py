"""Sandbox port readiness probe used during session bring-up.

Caido's HTTPS proxy takes a few seconds to start listening after the
Docker container is created. The session manager waits for it before
returning a session bundle so that the first tool call from an agent
doesn't hit a connection refused.

:func:`wait_for_tcp_ready` is the only probe — Caido serves an HTTP
forward proxy on its port and does *not* expose ``/health``. A TCP
connect is the most we can probe without sending real proxy traffic.
"""

from __future__ import annotations

import asyncio
import contextlib


class SandboxNotReadyError(Exception):
    """Raised when a sandbox port doesn't accept connections in time."""


# Default polling cadence between attempts. Balanced for CI-style
# fast bring-up (sub-second) without burning CPU when the port is
# legitimately taking a few seconds.
_DEFAULT_POLL_INTERVAL = 0.5


async def wait_for_tcp_ready(
    host: str,
    port: int,
    *,
    timeout: float = 30.0,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> None:
    """Poll ``host:port`` until a TCP connect succeeds, or raise after ``timeout``.

    Used for ports that don't expose an HTTP health endpoint (Caido's
    forward proxy). We open the socket and immediately close it — the
    handshake completing is enough to confirm readiness.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: str | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=poll_interval * 4,
            )
        except (TimeoutError, OSError) as e:
            last_error = type(e).__name__
        else:
            writer.close()
            # Some servers close hard immediately after accept; we only
            # care that the connect itself succeeded.
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            del reader
            return
        await asyncio.sleep(poll_interval)

    raise SandboxNotReadyError(
        f"TCP probe of {host}:{port} did not connect within {timeout}s (last error: {last_error})",
    )
