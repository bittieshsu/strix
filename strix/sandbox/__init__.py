"""Strix sandbox layer on top of OpenAI Agents SDK SandboxAgent / Manifest.

- :mod:`.caido_capability` — Caido proxy + 7 GraphQL function tools
  + system prompt block.
- :mod:`.healthcheck` — ``wait_for_ports_ready``.
- :mod:`.session_manager` — ``create_or_reuse`` / ``cleanup`` keyed
  by scan id.
"""
