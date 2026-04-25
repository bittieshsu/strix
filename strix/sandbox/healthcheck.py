"""Sandbox port readiness probes used during session bring-up.

The in-container tool server (FastAPI) takes a few seconds to start
listening after the Docker container is created, and Caido's HTTPS
proxy takes a similar window. The session manager waits for both
before returning a session bundle so that the first tool call from
an agent doesn't hit a connection refused.

Two helpers are exposed:

- :func:`wait_for_http_ready` for the FastAPI tool server, whose
  ``/health`` endpoint returns ``{"status": "healthy"}`` once the
  process is up. We don't require the JSON shape exactly — any 2xx
  is treated as ready.

- :func:`wait_for_tcp_ready` for Caido, which serves an HTTP forward
  proxy on its port and does *not* expose ``/health``. A TCP connect
  is the most we can probe without sending real proxy traffic.

References:
    - PLAYBOOK.md §3.1
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx


logger = logging.getLogger(__name__)


class SandboxNotReadyError(Exception):
    """Raised when a sandbox port doesn't accept connections in time."""


# Default per-attempt HTTP timeout. 5s so a slow first request (image
# still warming up) doesn't misfire as a hard failure on a single attempt.
_DEFAULT_HTTP_PROBE_TIMEOUT = 5.0

# Default polling cadence between attempts. Balanced for CI-style
# fast bring-up (sub-second) without burning CPU when the port is
# legitimately taking a few seconds.
_DEFAULT_POLL_INTERVAL = 0.5


async def wait_for_http_ready(
    url: str,
    *,
    timeout: float = 30.0,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    probe_timeout: float = _DEFAULT_HTTP_PROBE_TIMEOUT,
) -> None:
    """Poll ``url`` until any 2xx response, or raise after ``timeout``.

    Network errors (ConnectError / TimeoutException / RequestError)
    are treated as "not ready yet" — the loop continues. Any other
    exception class will surface immediately so a programmer error
    (bad URL, etc.) doesn't get silently retried for 30 seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=probe_timeout, trust_env=False) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await client.get(url)
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}"
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
                last_error = type(e).__name__
            await asyncio.sleep(poll_interval)

    raise SandboxNotReadyError(
        f"HTTP probe of {url} did not return 2xx within {timeout}s (last error: {last_error})",
    )


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
