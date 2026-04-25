"""Strix runtime package.

- :class:`strix.runtime.strix_docker_client.StrixDockerSandboxClient` —
  host-side ``DockerSandboxClient`` subclass that injects
  ``NET_ADMIN`` / ``NET_RAW`` capabilities and ``host.docker.internal``
  extra-hosts, used by the per-scan session manager
  (:mod:`strix.sandbox.session_manager`).

- ``tool_server.py`` — FastAPI server that runs inside the sandbox
  container. Sandbox-bound tools (browser, terminal, python, file_edit,
  proxy) POST here from the host via
  :func:`strix.tools._sandbox_dispatch.post_to_sandbox`.
"""
