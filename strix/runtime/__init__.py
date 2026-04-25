"""Strix runtime package.

What lives here:

- :class:`StrixDockerSandboxClient` — host-side ``DockerSandboxClient``
  subclass that injects ``NET_ADMIN`` / ``NET_RAW`` capabilities and
  ``host.docker.internal`` extra-hosts, used by the per-scan session
  manager (:mod:`strix.sandbox.session_manager`).

- ``tool_server.py`` — the FastAPI server that runs *inside* the
  sandbox container; sandbox-bound tools (browser, terminal, python,
  file_edit, proxy) POST here from the host via
  :func:`strix.tools._sandbox_dispatch.post_to_sandbox`.

The legacy DockerRuntime / AbstractRuntime + ``get_runtime`` /
``cleanup_runtime`` globals were removed when the SDK harness took
over scan lifecycle; sandbox sessions are now per-scan and managed by
:func:`strix.sandbox.session_manager.create_or_reuse`.
"""


class SandboxInitializationError(Exception):
    """Raised when sandbox initialization fails (e.g., Docker issues)."""

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


__all__ = ["SandboxInitializationError"]
