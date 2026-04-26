"""Multi-agent graph tools backed by the SDK-native :class:`AgentCoordinator`.

- ``view_agent_graph``: render the parent/child tree.
- ``agent_status``: per-agent status + pending message count.
- ``send_message_to_agent``: append a message to another agent's SDK session.
- ``wait_for_message``: pause this agent until a message arrives or
  ``timeout_seconds`` elapses.
- ``create_agent``: spawn a child via
  ``asyncio.create_task(Runner.run(...))``; the task handle is stored
  so a root-level cancel cascades to descendants.
- ``agent_finish``: subagents only — posts a structured completion
  report to the parent's SDK session and returns a final-output marker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agents import RunConfig, RunContextWrapper, function_tool
from agents.items import TResponseInputItem
from agents.model_settings import ModelSettings
from agents.sandbox import SandboxRunConfig

from strix.llm.multi_provider_setup import build_multi_provider
from strix.llm.retry import DEFAULT_RETRY
from strix.orchestration.coordinator import (
    coordinator_from_context,
    open_agent_session,
    run_with_continuation,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from agents import Agent as SDKAgent


logger = logging.getLogger(__name__)


def _ctx(ctx: RunContextWrapper) -> dict[str, Any]:
    return ctx.context if isinstance(ctx.context, dict) else {}


def _render_completion_report(
    *,
    agent_name: str,
    agent_id: str,
    task: str,
    success: bool,
    result_summary: str,
    findings: list[str],
    recommendations: list[str],
) -> str:
    """Render a child's completion report as plain structured text.

    Goes into the parent's SDK session with coordinator-added sender
    metadata, so this body just carries the contents. No XML — no
    escaping concerns, no parser ambiguity.
    """
    status = "SUCCESS" if success else "FAILED"
    completion_time = datetime.now(UTC).isoformat()

    lines: list[str] = [
        f"== Completion report from {agent_name} ({agent_id}) ==",
        f"Status: {status}",
        f"Time: {completion_time}",
    ]
    if task:
        lines.append(f"Task: {task}")
    lines.append("")
    lines.append("Summary:")
    lines.append(result_summary or "(none)")
    if findings:
        lines.append("")
        lines.append("Findings:")
        lines.extend(f"- {f}" for f in findings)
    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        lines.extend(f"- {r}" for r in recommendations)
    return "\n".join(lines)


@function_tool(timeout=30)
async def view_agent_graph(ctx: RunContextWrapper) -> str:
    """Print the multi-agent tree — every agent, its parent, its status.

    Use before spawning a new agent (don't duplicate work — check whether
    something specialized for that task already exists) and any time you
    want a snapshot of who's still ``running`` / ``waiting`` /
    ``completed`` / ``crashed`` / ``stopped``. Output is an indented
    bullet list with status in brackets; the agent that called this tool
    is marked ``← you``.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    me = inner.get("agent_id")
    if coordinator is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator not initialized in context."},
            ensure_ascii=False,
            default=str,
        )

    parent_of, statuses, names = await coordinator.graph_snapshot()

    lines: list[str] = []

    def render(aid: str, depth: int) -> None:
        status = statuses.get(aid, "?")
        marker = "  ← you" if aid == me else ""
        lines.append(f"{'  ' * depth}- {names.get(aid, aid)} ({aid}) [{status}]{marker}")
        for child, p in parent_of.items():
            if p == aid:
                render(child, depth + 1)

    roots = [aid for aid, parent in parent_of.items() if parent is None]
    for root in roots:
        render(root, 0)

    summary = {
        "total": len(parent_of),
        "running": sum(1 for s in statuses.values() if s == "running"),
        "waiting": sum(1 for s in statuses.values() if s == "waiting"),
        "completed": sum(1 for s in statuses.values() if s == "completed"),
        "crashed": sum(1 for s in statuses.values() if s == "crashed"),
        "stopped": sum(1 for s in statuses.values() if s == "stopped"),
    }
    return json.dumps(
        {
            "success": True,
            "graph_structure": "\n".join(lines) or "(no agents)",
            "summary": summary,
        },
        ensure_ascii=False,
        default=str,
    )


@function_tool(timeout=30)
async def agent_status(ctx: RunContextWrapper, agent_id: str) -> str:
    """Look up one agent's lifecycle state + pending message count.

    Use when you need precise state on a specific agent (e.g., "is the
    XSS specialist still going?") rather than the full tree view.
    Returns ``status`` (``running`` / ``waiting`` / ``completed`` /
    ``crashed`` / ``stopped``), ``parent_id``, and ``pending_messages``.

    Args:
        agent_id: The 8-char id from ``view_agent_graph`` /
            ``create_agent``.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    if coordinator is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator not initialized in context."},
            ensure_ascii=False,
            default=str,
        )

    info = await coordinator.agent_info(agent_id)
    if info is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown agent_id: {agent_id}",
            },
            ensure_ascii=False,
            default=str,
        )
    info["success"] = True
    return json.dumps(info, ensure_ascii=False, default=str)


@function_tool(timeout=30)
async def send_message_to_agent(
    ctx: RunContextWrapper,
    target_agent_id: str,
    message: str,
    message_type: Literal["query", "instruction", "information"] = "information",
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
) -> str:
    """Send a message to another agent's inbox — sparingly.

    Inter-agent messages are surfaced at the top of the target's next
    LLM turn. Use only when essential:

    - Sharing a discovered finding/credential another agent needs.
    - Asking a specialist a focused question.
    - Coordinating who covers what (avoid overlap).
    - Telling a child to wrap up or change course.

    **Don't** use for routine "hello/status" pings, for context the
    target already has (children inherit parent history), or when
    parent/child completion via ``agent_finish`` already covers the
    flow. Messages to any registered agent are queued, regardless of
    status, so a follow-up can wake a completed/stopped/failed agent.

    Args:
        target_agent_id: Recipient's 8-char id.
        message: The full message body. Be specific — include payloads,
            URLs, or what you want them to do, not just headlines.
        message_type: ``query`` (you want a reply), ``instruction``
            (you're directing them), ``information`` (FYI, no reply
            expected). Default ``information``.
        priority: ``low`` / ``normal`` / ``high`` / ``urgent``.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    me = inner.get("agent_id")
    if coordinator is None or me is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator or agent_id missing in context."},
            ensure_ascii=False,
            default=str,
        )

    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    delivered = await coordinator.send(
        target_agent_id,
        {
            "id": msg_id,
            "from": me,
            "content": message,
            "type": message_type,
            "priority": priority,
        },
    )
    if not delivered:
        return json.dumps(
            {
                "success": False,
                "error": f"Target agent '{target_agent_id}' not found or message delivery failed.",
            },
            ensure_ascii=False,
            default=str,
        )
    return json.dumps(
        {
            "success": True,
            "message_id": msg_id,
            "target_agent_id": target_agent_id,
            "delivery_status": "queued",
        },
        ensure_ascii=False,
        default=str,
    )


def _session_items_payload(items: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            payload.append({"role": role, "content": content})
        else:
            payload.append({"content": str(item)})
    return payload


@function_tool(timeout=601)
async def wait_for_message(
    ctx: RunContextWrapper,
    reason: str = "Waiting for messages from other agents",
    timeout_seconds: int = 600,
) -> str:
    """Pause this agent until a message lands in its inbox (or timeout).

    Use when you have nothing useful to do until a child/peer responds
    — typically after spawning subagents and you want to wait for
    their completion reports. The agent automatically resumes when any
    message arrives.

    **Critical caveats:**

    - **Never** call this if you finished your own task and have **no**
      child agents running — that's a permanent stall. Call
      ``finish_scan`` (root) or ``agent_finish`` (subagent) instead.
    - If you're waiting on an agent that **isn't your child**, message
      it first asking it to ping you when done — otherwise it has no
      reason to send to your inbox and you'll wait the full timeout.
    - Children update the parent automatically via ``agent_finish``
      → no extra coordination needed.

    Args:
        reason: One-line note shown in graph snapshots while you're
            waiting (helps a human or sibling agent debug who's stuck
            on what).
        timeout_seconds: Hard cap (default 600s). On timeout the tool
            returns and you decide whether to keep working or wait
            again.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    me = inner.get("agent_id")
    interactive = bool(inner.get("interactive", False))
    if coordinator is None or me is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator or agent_id missing in context."},
            ensure_ascii=False,
            default=str,
        )

    async with coordinator._lock:
        stopped = coordinator.statuses.get(me) == "stopped"
    if stopped:
        return json.dumps(
            {
                "success": True,
                "status": "stopped",
                "reason": reason,
                "note": "Wait ended because this agent is stopped.",
            },
            ensure_ascii=False,
            default=str,
        )

    pending = await coordinator.pending_count(me)
    if pending > 0:
        items = await coordinator.recent_session_items(me, pending)
        await coordinator.consume_wake(me)
        await coordinator.mark_running(me)
        return json.dumps(
            {
                "success": True,
                "status": "message_arrived",
                "pending_messages": pending,
                "messages": _session_items_payload(items),
                "reason": reason,
            },
            ensure_ascii=False,
            default=str,
        )

    if interactive:
        await coordinator.park_waiting(me)
        inner["agent_waiting_called"] = True
        return json.dumps(
            {
                "success": True,
                "status": "waiting",
                "agent_waiting": True,
                "reason": reason,
                "note": "Agent parked; execution will resume when a message arrives.",
            },
            ensure_ascii=False,
            default=str,
        )

    await coordinator.park_waiting(me)
    try:
        await asyncio.wait_for(coordinator.wait_for_message(me), timeout_seconds)
    except TimeoutError:
        await coordinator.mark_running(me)
        return json.dumps(
            {
                "success": True,
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "reason": reason,
                "note": "No messages within timeout — continue work or call agent_finish.",
            },
            ensure_ascii=False,
            default=str,
        )

    async with coordinator._lock:
        stopped = coordinator.statuses.get(me) == "stopped"
    if stopped:
        return json.dumps(
            {
                "success": True,
                "status": "stopped",
                "reason": reason,
                "note": "Wait ended because this agent is stopped.",
            },
            ensure_ascii=False,
            default=str,
        )

    pending = await coordinator.pending_count(me)
    items = await coordinator.recent_session_items(me, pending)
    await coordinator.consume_wake(me)
    await coordinator.mark_running(me)

    return json.dumps(
        {
            "success": True,
            "status": "message_arrived",
            "pending_messages": pending,
            "messages": _session_items_payload(items),
            "reason": reason,
        },
        ensure_ascii=False,
        default=str,
    )


@function_tool(timeout=120)
async def create_agent(
    ctx: RunContextWrapper,
    name: str,
    task: str,
    inherit_context: bool = True,
    skills: list[str] | None = None,
) -> str:
    """Spawn a specialist child agent to run in parallel.

    Decompose complex pentests by handing focused subtasks to dedicated
    children. The child runs asynchronously — the parent continues
    immediately and can ``wait_for_message`` later (or just keep
    working in parallel). When the child calls ``agent_finish``, its
    completion report lands in the parent's inbox.

    **Before spawning, call ``view_agent_graph``** to confirm no
    existing agent already covers this scope — duplicate specialists
    waste turns and create coordination headaches.

    **Specialization principles:**

    - Most agents need at least one ``skill`` to be useful.
    - Aim for **1-3 related skills** per agent. Up to 5 only when the
      task genuinely spans them.
    - One skill = most focused (e.g., XSS-only). Five skills = upper
      bound.
    - Match the ``name`` to the focus (``XSS Specialist``,
      ``SQLi Validator``, ``Auth Specialist``).

    **When to spawn vs do it yourself:**

    - Spawn when the subtask is large, parallelizable, or needs
      different specialization than what you're already doing.
    - Don't spawn for trivial one-shot probes — just run the tool
      yourself.

    Args:
        name: Human-readable child name (used in graph views and
            ``send_message_to_agent`` flows).
        task: Specific objective. Be concrete — what to test, what
            success looks like, any constraints.
        inherit_context: Default ``True``. The child receives the
            parent's input history as background; only set ``False``
            when starting a clean-slate task.
        skills: Comma-separated skill names. Max 5; prefer 1-3.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    parent_id = inner.get("agent_id")
    factory: Callable[..., SDKAgent] | None = inner.get("agent_factory")

    if coordinator is None or parent_id is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator or agent_id missing in context."},
            ensure_ascii=False,
            default=str,
        )
    if factory is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No agent_factory in context. "
                    "The root assembly must inject one when building the run context."
                ),
            },
            ensure_ascii=False,
            default=str,
        )

    child_id = uuid.uuid4().hex[:8]

    try:
        child_agent = factory(name=name, skills=skills or [])
    except Exception as e:
        logger.exception("agent_factory raised while building child '%s'", name)
        return json.dumps(
            {
                "success": False,
                "error": f"agent_factory failed: {e!s}",
            },
            ensure_ascii=False,
            default=str,
        )

    await coordinator.register(
        child_id,
        name,
        parent_id,
        task=task,
        skills=list(skills or []),
        is_whitebox=bool(inner.get("is_whitebox", False)),
        scan_mode=str(inner.get("scan_mode", "deep")),
        diff_scope=inner.get("diff_scope"),
    )

    # ``ctx.turn_input`` carries the parent's full conversation up to and
    # including the call that's currently invoking ``create_agent``
    # (populated by SDK at ``run_internal/turn_resolution.py:806``).
    # Wrap as a single read-only block so the child sees the parent's
    # reasoning as background but doesn't try to continue parent's turns.
    parent_history = list(ctx.turn_input) if inherit_context and ctx.turn_input else []
    initial_input: list[TResponseInputItem] = []
    if parent_history:
        rendered = json.dumps(parent_history, ensure_ascii=False, default=str)
        initial_input.append(
            {
                "role": "user",
                "content": (
                    "== Inherited context from parent (background only) ==\n"
                    f"{rendered}\n"
                    "== End of inherited context ==\n"
                    "Use the above as background only; do not continue the "
                    "parent's work. Your task follows."
                ),
            },
        )
    initial_input.append(
        {
            "role": "user",
            "content": (
                f"You are agent {name} ({child_id}); your parent is {parent_id}. "
                f"Maintain your own identity. Call agent_finish when your task "
                f"is complete."
            ),
        }
    )
    initial_input.append({"role": "user", "content": task})

    child_ctx: dict[str, Any] = {
        "coordinator": coordinator,
        "sandbox_session": inner.get("sandbox_session"),
        "sandbox_client": inner.get("sandbox_client"),
        "caido_client": inner.get("caido_client"),
        "agent_id": child_id,
        "parent_id": parent_id,
        "tracer": inner.get("tracer"),
        "model": inner["model"],
        "model_settings": inner.get("model_settings"),
        "max_turns": int(inner.get("max_turns", 300)),
        "agent_finish_called": False,
        "is_whitebox": bool(inner.get("is_whitebox", False)),
        "interactive": bool(inner.get("interactive", False)),
        "scan_mode": str(inner.get("scan_mode", "deep")),
        "diff_scope": inner.get("diff_scope"),
        "run_id": inner.get("run_id"),
        "agent_factory": factory,
        # Stashed for ``agent_finish`` to echo back in its completion report.
        "task": task,
        # Inherited so ``create_agent`` calls from this child can also
        # add their per-child sessions to the same teardown list.
        "_sessions_to_close": inner.get("_sessions_to_close", []),
    }

    # Every agent gets its own SDK session_id inside the shared agents.db.
    agents_db_path = inner.get("agents_db_path")
    if not isinstance(agents_db_path, Path):
        run_id = inner.get("run_id") or "default"
        agents_db_path = Path.cwd() / "strix_runs" / str(run_id) / "agents.db"
    child_session = open_agent_session(child_id, agents_db_path)
    sessions_list = child_ctx.get("_sessions_to_close")
    if isinstance(sessions_list, list):
        sessions_list.append(child_session)
    await coordinator.attach_runtime(child_id, session=child_session)

    child_model_settings = ModelSettings(
        parallel_tool_calls=False,
        tool_choice="required",
        retry=DEFAULT_RETRY,
    )
    override = inner.get("model_settings")
    if override is not None:
        child_model_settings = child_model_settings.resolve(override)
    sandbox_session = inner.get("sandbox_session")
    child_run_config = RunConfig(
        model=inner["model"],
        model_provider=build_multi_provider(),
        model_settings=child_model_settings,
        sandbox=(
            SandboxRunConfig(client=inner.get("sandbox_client"), session=sandbox_session)
            if sandbox_session is not None
            else None
        ),
        tracing_disabled=False,
        trace_include_sensitive_data=False,
    )

    task_handle = asyncio.create_task(
        run_with_continuation(
            agent=child_agent,
            initial_input=initial_input,
            run_config=child_run_config,
            context=child_ctx,
            max_turns=int(inner.get("max_turns", 300)),
            coordinator=coordinator,
            agent_id=child_id,
            interactive=bool(inner.get("interactive", False)),
            session=child_session,
        ),
        name=f"agent-{name}-{child_id}",
    )
    await coordinator.attach_runtime(child_id, task=task_handle)

    logger.info(
        "create_agent: spawned %s (%s) parent=%s skills=%d task_len=%d",
        child_id,
        name,
        parent_id or "-",
        len(skills or []),
        len(task or ""),
    )

    return json.dumps(
        {
            "success": True,
            "agent_id": child_id,
            "name": name,
            "parent_id": parent_id,
            "message": f"Spawned '{name}' ({child_id}) running in parallel.",
        },
        ensure_ascii=False,
        default=str,
    )


@function_tool(timeout=30)
async def agent_finish(
    ctx: RunContextWrapper,
    result_summary: str,
    findings: list[str] | None = None,
    success: bool = True,
    report_to_parent: bool = True,
    final_recommendations: list[str] | None = None,
) -> str:
    """Subagent termination — post a completion report to the parent.

    **Subagents only.** Root agents must call ``finish_scan`` instead;
    this tool refuses to run for root agents. Calling this:

    1. Marks the subagent as ``completed``.
    2. Posts a structured completion report to the parent's inbox
       (when ``report_to_parent`` is true).
    3. Stops this subagent's execution.

    **Vulnerability findings must already be filed via
    ``create_vulnerability_report`` before calling this.** The
    ``findings`` field here is for narrative summary only — it does
    not register vulns in the scan report.

    Write the summary as if the parent has no idea what you were
    doing: what did you test, what did you find/confirm/rule out,
    what's still open.

    Args:
        result_summary: What you accomplished and discovered. Concrete
            and specific (URLs, parameters, payloads that worked).
        findings: Optional bullet list of confirmed observations. For
            credit-bearing vulnerabilities, file
            ``create_vulnerability_report`` first; this is for
            narrative.
        success: Whether the assigned subtask was completed
            successfully. Default ``True``.
        report_to_parent: Whether to deliver the completion report to
            the parent's inbox. Default ``True``.
        final_recommendations: Optional next-step suggestions for the
            parent (e.g., "prioritize testing X", "spawn an agent to
            cover Y").
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    me = inner.get("agent_id")
    if coordinator is None or me is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator or agent_id missing in context."},
            ensure_ascii=False,
            default=str,
        )

    parent_id = inner.get("parent_id")
    if parent_id is None:
        return json.dumps(
            {
                "success": False,
                "agent_completed": False,
                "error": (
                    "agent_finish is for subagents. Root/main agents must call finish_scan instead."
                ),
                "parent_notified": False,
            },
            ensure_ascii=False,
            default=str,
        )

    # The coordinator settles lifecycle from this JSON final output after
    # the SDK run finishes; the tool only sends the parent report.
    inner["agent_finish_called"] = True

    parent_notified = False
    if report_to_parent:
        async with coordinator._lock:
            agent_name = coordinator.names.get(me, me)
        report = _render_completion_report(
            agent_name=agent_name,
            agent_id=me,
            task=str(inner.get("task", "")),
            success=success,
            result_summary=result_summary,
            findings=list(findings or []),
            recommendations=list(final_recommendations or []),
        )
        await coordinator.send(
            parent_id,
            {
                "id": f"report_{uuid.uuid4().hex[:8]}",
                "from": me,
                "content": report,
                "type": "completion",
                "priority": "high",
            },
        )
        parent_notified = True

    logger.info(
        "agent_finish: %s success=%s findings=%d parent_notified=%s",
        me,
        success,
        len(findings or []),
        parent_notified,
    )

    return json.dumps(
        {
            "success": True,
            "agent_completed": True,
            "parent_notified": parent_notified,
            "agent_id": me,
            "summary": result_summary,
            "findings_count": len(findings or []),
            "has_recommendations": bool(final_recommendations),
        },
        ensure_ascii=False,
        default=str,
    )


@function_tool(timeout=30)
async def stop_agent(
    ctx: RunContextWrapper,
    target_agent_id: str,
    cascade: bool = True,
    reason: str = "",
) -> str:
    """Gracefully stop a running agent (and optionally its descendants).

    Uses the SDK's ``RunResultStreaming.cancel(mode="after_turn")`` so the
    target's current turn finishes — including saving items to its
    session — before the run loop honors the cancel. The agent's
    interactive outer loop parks as ``stopped``; later user/peer
    messages can wake it again.

    Use sparingly. Prefer ``send_message_to_agent`` (asking the agent
    to wrap up) for soft-stop scenarios. Reach for ``stop_agent`` when
    a child has gone off-track and won't self-correct.

    Args:
        target_agent_id: The 8-char id from ``view_agent_graph`` /
            ``create_agent``. Cannot stop yourself.
        cascade: If ``True`` (default), also stop every descendant of
            ``target_agent_id`` leaves-first. ``False`` stops only the
            target.
        reason: Optional human-readable reason for the stop, surfaced
            in logs and telemetry.
    """
    inner = _ctx(ctx)
    coordinator = coordinator_from_context(inner)
    me = inner.get("agent_id")
    if coordinator is None or me is None:
        return json.dumps(
            {"success": False, "error": "Agent coordinator or agent_id missing in context."},
            ensure_ascii=False,
            default=str,
        )
    if target_agent_id == me:
        return json.dumps(
            {
                "success": False,
                "error": "Cannot stop yourself; call agent_finish or finish_scan instead.",
            },
            ensure_ascii=False,
            default=str,
        )
    if await coordinator.agent_info(target_agent_id) is None:
        return json.dumps(
            {"success": False, "error": f"Unknown agent_id: {target_agent_id}"},
            ensure_ascii=False,
            default=str,
        )

    if cascade:
        await coordinator.cancel_descendants_graceful(target_agent_id)
    else:
        await coordinator.request_stop(target_agent_id)

    logger.info(
        "stop_agent: target=%s cascade=%s reason=%r",
        target_agent_id,
        cascade,
        reason,
    )
    return json.dumps(
        {
            "success": True,
            "target_agent_id": target_agent_id,
            "cascade": cascade,
            "reason": reason,
            "note": "Cancellation is graceful — current turn completes first.",
        },
        ensure_ascii=False,
        default=str,
    )
