"""``make_run_config`` — assemble a Strix-flavored ``RunConfig`` for ``Runner.run``.

Every scan goes through here so defaults apply uniformly. Per-call
overrides land via ``model_settings_override``.
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


# Sequential tool calls per agent — the tool server serializes one task
# per agent at a time, so concurrent calls would queue anyway.
_PARALLEL_TOOL_CALLS_DEFAULT = False

# Retry policy. 401/403/400 are deliberately excluded — auth and
# validation errors can't be fixed by retrying and should fail fast.
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
    model: str = "anthropic/claude-sonnet-4-6",
    parallel_tool_calls: bool = _PARALLEL_TOOL_CALLS_DEFAULT,
    tool_choice: Literal["auto", "required", "none"] | None = "required",
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    model_settings_override: ModelSettings | None = None,
    sandbox_client: Any | None = None,
) -> RunConfig:
    """Build a ``RunConfig`` with Strix defaults.

    Note: ``max_turns`` is not a ``RunConfig`` field — pass it directly
    to ``Runner.run``. ``STRIX_DEFAULT_MAX_TURNS`` is the budget Strix
    uses.

    Args:
        sandbox_session: Live sandbox session shared by every agent in
            this scan (one container per scan; see
            :mod:`strix.sandbox.session_manager`). ``None`` is allowed
            for unit tests and dry runs.
        model: Model alias passed to ``MultiProvider``. Defaults to the
            production Anthropic alias.
        parallel_tool_calls: Default ``False`` — the tool server
            serializes one task per agent.
        tool_choice: Forces tool use per turn unless explicitly relaxed.
        reasoning_effort: ``"low" | "medium" | "high"``; routes to
            ``ModelSettings.reasoning``.
        model_settings_override: Optional per-run ``ModelSettings``
            merged over factory defaults.
        sandbox_client: Optional pre-built sandbox client (Strix Docker
            subclass). The SDK instantiates its built-in if a session is
            supplied without a client.
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
    model: str = "anthropic/claude-sonnet-4-6",
    model_settings: ModelSettings | None = None,
    max_turns: int = 300,
    is_whitebox: bool = False,
    diff_scope: dict[str, Any] | None = None,
    run_id: str | None = None,
    sandbox_client: Any | None = None,
    agent_factory: Any | None = None,
    caido_capability: Any | None = None,
) -> dict[str, Any]:
    """Build the per-agent ``context`` dict passed to ``Runner.run(context=...)``.

    The canonical place where bus, sandbox handles, identity, tracer
    reference, and per-agent toggles live. Tools, hooks, and
    ``inject_messages_filter`` reach in via ``ctx.context.get(...)``.

    ``agent_factory`` is a callable ``(name, skills) -> agents.Agent`` —
    the ``create_agent`` graph tool uses it to spin up children that
    inherit the same wiring. ``sandbox_client`` is the host-side Docker
    subclass, reused across child runs.
    """
    return {
        "bus": bus,
        "sandbox_session": sandbox_session,
        "sandbox_client": sandbox_client,
        "sandbox_token": sandbox_token,
        "tool_server_host_port": tool_server_host_port,
        "caido_host_port": caido_host_port,
        "caido_capability": caido_capability,
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
