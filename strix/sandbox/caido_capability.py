"""CaidoCapability — sandbox capability for the Caido HTTP/HTTPS proxy.

Three concerns wired into the SDK's capability lifecycle:

1. **Manifest mutation** (``process_manifest``): inject ``http_proxy`` /
   ``https_proxy`` / ``ALL_PROXY`` env vars pointing at the in-container
   Caido listener. Any tool that ultimately shells out (curl, requests,
   etc.) now flows through the proxy automatically.

2. **Tool exposure** (``tools``): the seven Caido SDK function-tool
   wrappers are returned here. The SDK runtime collects tools from
   every capability and merges them with the agent's ``tools=[...]``
   declaration, so agents don't have to redeclare them.

3. **Healthcheck task** (``bind``): when a session binds, we kick off
   :func:`wait_for_http_ready` against the FastAPI tool server's
   ``/health`` endpoint and :func:`wait_for_tcp_ready` against the
   Caido proxy port. The aggregated task handle is stored on
   ``self._healthcheck_task``, which the
   :class:`StrixOrchestrationHooks.on_agent_start` hook awaits before
   the first LLM call so the agent never hits a connection-refused
   on its very first tool invocation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from agents.sandbox.capabilities.capability import Capability
from agents.tool import Tool
from pydantic import PrivateAttr

from strix.sandbox.healthcheck import wait_for_http_ready, wait_for_tcp_ready
from strix.tools.proxy.tools import (
    list_requests,
    list_sitemap,
    repeat_request,
    scope_rules,
    send_request,
    view_request,
    view_sitemap_entry,
)


if TYPE_CHECKING:
    from agents.sandbox.manifest import Manifest
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession


logger = logging.getLogger(__name__)


# Container-internal Caido listener. The in-container Caido sidecar binds
# on this port; the host gets a randomly mapped port we resolve at
# session create time and pass into the per-agent context as
# ``caido_host_port`` for the proxy SDK tools' dispatcher.
_CAIDO_INTERNAL_PORT = 48080

# Container-internal FastAPI tool server. Same shape as Caido — host
# port is resolved at session create.
_TOOL_SERVER_INTERNAL_PORT = 48081

# Probe URLs used inside ``bind``. ``host=127.0.0.1`` because the host
# port mapping is loopback-only.
_PROBE_HOST = "127.0.0.1"


# Cached tool list — building Tool instances has side effects via the
# function_tool decorator and we don't want re-instantiation each time
# the SDK calls ``tools()``.
_CAIDO_TOOLS: tuple[Tool, ...] = (
    list_requests,
    view_request,
    send_request,
    repeat_request,
    scope_rules,
    list_sitemap,
    view_sitemap_entry,
)


class CaidoCapability(Capability):
    """Caido HTTP/HTTPS forward proxy + 7 GraphQL function tools.

    Lifetime: one instance per scan. The SDK clones capabilities
    per-run (see ``Capability.clone``); we accept that — each cloned
    instance opens its own healthcheck task on ``bind``, which is
    cheap and idempotent.
    """

    type: Literal["caido"] = "caido"

    # Pydantic ``PrivateAttr`` for runtime-only state. Pydantic forbids
    # underscore-prefixed *fields*, but private attributes are first-class
    # and cleanly excluded from model dumps and serialization.
    _healthcheck_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    # The two ports the host needs to reach. Populated by the session
    # manager *after* the SDK creates the container and we've resolved
    # the random host-side mappings via ``session._resolve_exposed_port``.
    _tool_server_host_port: int | None = PrivateAttr(default=None)
    _caido_host_port: int | None = PrivateAttr(default=None)

    # Per-capability healthcheck timeout. Long enough to cover image
    # pulls on a cold cache plus tool-server boot, short enough that a
    # mis-configured image fails the run inside a few minutes.
    _HEALTHCHECK_TIMEOUT: ClassVar[float] = 60.0

    def process_manifest(self, manifest: Manifest) -> Manifest:
        """Inject proxy env vars into the manifest's environment.

        Mutates in place; returns the same manifest. Mirrors the SDK's
        Capability protocol where ``process_manifest`` is the single
        synchronous hook for changing what the container sees.
        """
        env = dict(manifest.environment.value or {})
        env.update(
            {
                "http_proxy": f"http://127.0.0.1:{_CAIDO_INTERNAL_PORT}",
                "https_proxy": f"http://127.0.0.1:{_CAIDO_INTERNAL_PORT}",
                "ALL_PROXY": f"http://127.0.0.1:{_CAIDO_INTERNAL_PORT}",
            },
        )
        manifest.environment.value = env
        return manifest

    def tools(self) -> list[Tool]:
        """Return the seven Caido function tools.

        The SDK runtime calls this at agent-build time and merges the
        result with the agent's own tool list. Returning a fresh list
        each call (rather than yielding the cached tuple directly) is
        SDK convention.
        """
        return list(_CAIDO_TOOLS)

    async def instructions(self, manifest: Manifest) -> str | None:  # noqa: ARG002
        """System-prompt fragment appended for every Caido-equipped agent."""
        return (
            "<caido_proxy>\n"
            "All HTTP/HTTPS traffic in this sandbox is automatically captured "
            f"by Caido (in-container at 127.0.0.1:{_CAIDO_INTERNAL_PORT}; "
            "host_proxy / https_proxy env vars are pre-set).\n"
            "Tools: list_requests, view_request, send_request, repeat_request, "
            "scope_rules, list_sitemap, view_sitemap_entry.\n"
            "HTTPQL filter examples: "
            "'request.method == \"POST\"', "
            "'response.status >= 400', "
            "'request.host == \"target.com\"'.\n"
            "</caido_proxy>"
        )

    def configure_host_ports(
        self,
        *,
        tool_server_host_port: int,
        caido_host_port: int,
    ) -> None:
        """Record the resolved host-side ports.

        Called by the session manager after ``client.create(...)``
        returns, before binding the session. The healthcheck task
        reads these to know which mapped ports to probe.
        """
        self._tool_server_host_port = tool_server_host_port
        self._caido_host_port = caido_host_port

    def bind(self, session: BaseSandboxSession) -> None:
        """Schedule a healthcheck task on session bind.

        Stores the task handle so :class:`StrixOrchestrationHooks` can
        await it on the first agent start. We never raise from here —
        the healthcheck failure surfaces inside on_agent_start, which
        is the right place to fail the run because by then we have a
        live RunContextWrapper to log against.
        """
        super().bind(session)
        if self._tool_server_host_port is None or self._caido_host_port is None:
            logger.warning(
                "CaidoCapability.bind called before configure_host_ports; "
                "skipping healthcheck task scheduling.",
            )
            return
        self._healthcheck_task = asyncio.create_task(
            self._run_healthcheck(),
            name=f"caido-healthcheck-{self._tool_server_host_port}",
        )

    async def _run_healthcheck(self) -> None:
        """Probe both ports concurrently; raise on first failure."""
        # Mypy sees these as Optional, but ``bind`` checks both before
        # creating the task.
        assert self._tool_server_host_port is not None
        assert self._caido_host_port is not None
        await asyncio.gather(
            wait_for_http_ready(
                f"http://{_PROBE_HOST}:{self._tool_server_host_port}/health",
                timeout=self._HEALTHCHECK_TIMEOUT,
            ),
            wait_for_tcp_ready(
                _PROBE_HOST,
                self._caido_host_port,
                timeout=self._HEALTHCHECK_TIMEOUT,
            ),
        )
