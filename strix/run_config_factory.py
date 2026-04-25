"""make_run_config — assemble a Strix-flavored ``RunConfig`` for ``Runner.run``.

Factory pattern: every Strix scan goes through here so the defaults are
applied uniformly. Per-call overrides are accepted via ``model_settings_override``
for the rare case a single run wants different reasoning effort or
``tool_choice`` (C21).

References:
    - PLAYBOOK.md §2.10
    - AUDIT.md §2.1 (C1 — parallel_tool_calls=False until Phase 6 relaxes the
      tool server's per-agent task slot serialization)
    - AUDIT_R2.md §1.6 (C11 — retry policy explicitly excludes 401/403/400;
      auth and validation errors must fail fast, not waste retries)
    - AUDIT_R3.md C21 — RunConfig override + context fields including
      ``is_whitebox``, ``diff_scope``, ``run_id``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from agents import RunConfig
from agents.model_settings import ModelSettings
from agents.retry import (
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    retry_policies,
)
from agents.sandbox import SandboxRunConfig
from openai.types.shared import Reasoning

from strix.llm.multi_provider_setup import build_multi_provider
from strix.orchestration.filter import inject_messages_filter


if TYPE_CHECKING:
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession

    from strix.orchestration.bus import AgentMessageBus


# Phase 6 relaxes the tool server's per-agent task-slot serialization
# (``runtime/tool_server.py:94-97``) and flips this to ``True`` after
# multi-agent stress tests confirm safety.
_PHASE1_PARALLEL_DEFAULT = False

# Default retry policy. Explicitly does NOT include 401/403/400 — those are
# auth and validation errors that retrying cannot fix; they should fail fast
# so the user sees the real error within seconds. 429/5xx is the right set.
_RETRYABLE_HTTP_STATUSES = (429, 500, 502, 503, 504)

# Default retry budget: 5 attempts with ``min(90, 2*2^n)`` backoff.
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF = ModelRetryBackoffSettings(
    initial_delay=2.0,
    max_delay=90.0,
    multiplier=2.0,
    jitter=False,
)


def _default_retry_policy() -> Any:
    """Build the default retry policy.

    Built from ``retry_policies.any(...)``: any of the listed conditions
    triggers a retry. ``provider_suggested`` honors server-sent
    ``Retry-After`` hints; ``network_error`` covers connection / timeout;
    ``http_status`` whitelists transient HTTP codes.
    """
    return retry_policies.any(
        retry_policies.provider_suggested(),
        retry_policies.network_error(),
        retry_policies.http_status(_RETRYABLE_HTTP_STATUSES),
    )


#: Default ``max_turns`` callers should pass to ``Runner.run``.
STRIX_DEFAULT_MAX_TURNS = 300


def make_run_config(
    *,
    sandbox_session: BaseSandboxSession | None,
    model: str = "strix/claude-sonnet-4.6",
    parallel_tool_calls: bool = _PHASE1_PARALLEL_DEFAULT,
    tool_choice: Literal["auto", "required", "none"] | None = "required",
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    model_settings_override: ModelSettings | None = None,
    sandbox_client: Any | None = None,
) -> RunConfig:
    """Build a ``RunConfig`` with Strix defaults.

    Note: ``max_turns`` and ``isolate_parallel_failures`` are NOT
    ``RunConfig`` fields — they are passed directly to ``Runner.run``.
    Use ``STRIX_DEFAULT_MAX_TURNS`` for the budget; pass
    ``isolate_parallel_failures=False`` to ``Runner.run`` if Phase 6 has
    not yet flipped ``parallel_tool_calls=True``.

    Args:
        sandbox_session: Live sandbox session shared by every agent in this
            scan (one container per scan; see ``strix.sandbox.session_manager``).
            ``None`` is allowed for unit tests and dry runs.
        model: Model alias to pass to ``MultiProvider``. Defaults to the
            current production-favored Anthropic alias.
        parallel_tool_calls: Default ``False`` to keep behavior sequential
            per the tool server's slot serialization (C1).
        tool_choice: Forces tool use per turn unless explicitly relaxed.
            Pass ``None`` to omit.
        reasoning_effort: ``"low" | "medium" | "high"``; routes to
            ``ModelSettings.reasoning``. ``None`` defers to provider default.
        model_settings_override: Optional ``ModelSettings`` to merge over
            the factory defaults (C21 — per-run override path).
        sandbox_client: Optional pre-built sandbox client (e.g., the Strix
            Docker subclass). Defaults to ``None``; the SDK will instantiate
            its built-in if a session is supplied without a client.

    Returns:
        A ``RunConfig`` ready to pass to ``Runner.run``.
    """
    base_settings = ModelSettings(
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        retry=ModelRetrySettings(
            max_retries=_DEFAULT_MAX_RETRIES,
            backoff=_DEFAULT_BACKOFF,
            policy=_default_retry_policy(),
        ),
    )
    if reasoning_effort is not None:
        base_settings = base_settings.resolve(
            ModelSettings(reasoning=Reasoning(effort=reasoning_effort)),
        )
    if model_settings_override is not None:
        # ``ModelSettings.resolve`` merges another ModelSettings into self
        # with override-wins semantics — exactly what we want.
        base_settings = base_settings.resolve(model_settings_override)

    sandbox_config = (
        SandboxRunConfig(client=sandbox_client, session=sandbox_session)
        if sandbox_session is not None
        else None
    )

    return RunConfig(
        model=model,
        model_provider=build_multi_provider(),
        model_settings=base_settings,
        sandbox=sandbox_config,
        call_model_input_filter=inject_messages_filter,
        tracing_disabled=False,
        trace_include_sensitive_data=False,
    )


def make_agent_context(
    *,
    bus: AgentMessageBus,
    sandbox_session: BaseSandboxSession | None,
    sandbox_token: str | None,
    tool_server_host_port: int | None,
    caido_host_port: int | None,
    agent_id: str,
    agent_name: str,
    parent_id: str | None,
    tracer: Any | None,
    model: str = "strix/claude-sonnet-4.6",
    model_settings: ModelSettings | None = None,
    max_turns: int = 300,
    is_whitebox: bool = False,
    diff_scope: dict[str, Any] | None = None,
    run_id: str | None = None,
    sandbox_client: Any | None = None,
    agent_factory: Any | None = None,
) -> dict[str, Any]:
    """Build the per-agent ``context`` dict passed to ``Runner.run(context=...)``.

    The dict is the canonical place where bus, sandbox handles, identity,
    tracer reference, and per-agent toggles live. Tools, hooks, and the
    ``inject_messages_filter`` all reach in via ``ctx.context.get(...)``.

    ``agent_factory`` is a callable ``(name, skills) -> agents.Agent`` used by
    the ``create_agent`` graph tool to spin up children. ``sandbox_client``
    is the host-side Docker subclass; ``create_agent`` reuses it across
    child runs.
    """
    return {
        "bus": bus,
        "sandbox_session": sandbox_session,
        "sandbox_client": sandbox_client,
        "sandbox_token": sandbox_token,
        "tool_server_host_port": tool_server_host_port,
        "caido_host_port": caido_host_port,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "parent_id": parent_id,
        "tracer": tracer,
        "model": model,
        "model_settings": model_settings,
        "max_turns": max_turns,
        "turn_count": 0,
        "agent_finish_called": False,
        "is_whitebox": is_whitebox,
        "diff_scope": diff_scope,
        "run_id": run_id,
        "agent_factory": agent_factory,
    }
