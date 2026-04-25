"""SDK function-tool wrappers for the multi-agent graph tools.

Six tools that read/write the :class:`AgentMessageBus` (built in Phase 0,
``strix.orchestration.bus``):

- ``view_agent_graph``: render the parent/child tree from ``bus.parent_of``.
- ``agent_status``: per-agent status + pending message count.
- ``send_message_to_agent``: peer-to-peer message into a child/sibling inbox.
- ``wait_for_message``: poll our own inbox until a message arrives or the
  timeout expires (the legacy harness's "I'm idle, wake me on inbox").
- ``create_agent``: spawn a child via ``asyncio.create_task(Runner.run(...))``;
  registers the child with the bus and stores its task handle so root cancels
  cascade (C9, ``bus.cancel_descendants``).
- ``agent_finish``: subagents only — flips ``agent_finish_called`` so the
  on_agent_end hook records "completed" rather than "crashed" (C8), and
  posts a structured completion report to the parent's inbox.

The legacy ``strix.tools.agents_graph.agents_graph_actions`` is left
untouched — it still drives the legacy harness. These wrappers only
target the bus and don't touch the legacy ``_agent_graph`` dict.

References:
    - PLAYBOOK.md §4.3
    - AUDIT_R2.md §1.4 (cancel_descendants — Runner.run task handle stored
      in bus.tasks so a root cancel walks the tree)
    - AUDIT_R3.md C8 (crash detection via on_agent_end + agent_finish_called)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Literal

from agents import RunContextWrapper, Runner
from agents.items import TResponseInputItem

from strix.orchestration.hooks import StrixOrchestrationHooks
from strix.run_config_factory import make_agent_context, make_run_config
from strix.tools._decorator import strix_tool


if TYPE_CHECKING:
    from collections.abc import Callable

    from agents import Agent as SDKAgent


logger = logging.getLogger(__name__)


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=30)
async def view_agent_graph(ctx: RunContextWrapper) -> str:
    """Render the multi-agent tree starting from each root.

    Output is a single string the model can parse: indented bullet list,
    one line per agent, status in brackets. Roots are agents whose
    ``parent_of[id]`` is ``None``.
    """
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    me = inner.get("agent_id")
    if bus is None:
        return _dump({"success": False, "error": "Bus not initialized in context."})

    async with bus._lock:
        parent_of = dict(bus.parent_of)
        statuses = dict(bus.statuses)
        names = dict(bus.names)

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
    return _dump(
        {
            "success": True,
            "graph_structure": "\n".join(lines) or "(no agents)",
            "summary": summary,
        }
    )


@strix_tool(timeout=30)
async def agent_status(ctx: RunContextWrapper, agent_id: str) -> str:
    """Inspect one agent's lifecycle state and pending message count."""
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    if bus is None:
        return _dump({"success": False, "error": "Bus not initialized in context."})

    async with bus._lock:
        if agent_id not in bus.statuses:
            return _dump(
                {
                    "success": False,
                    "error": f"Unknown agent_id: {agent_id}",
                }
            )
        return _dump(
            {
                "success": True,
                "agent_id": agent_id,
                "name": bus.names.get(agent_id),
                "status": bus.statuses.get(agent_id),
                "parent_id": bus.parent_of.get(agent_id),
                "pending_messages": len(bus.inboxes.get(agent_id, [])),
            }
        )


@strix_tool(timeout=30)
async def send_message_to_agent(
    ctx: RunContextWrapper,
    target_agent_id: str,
    message: str,
    message_type: Literal["query", "instruction", "information"] = "information",
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
) -> str:
    """Queue a message for another agent's inbox.

    The target's next ``inject_messages_filter`` pass (top of its next LLM
    turn) drains the inbox and surfaces the message wrapped in
    ``<inter_agent_message>``. Messages to a finalized agent are dropped
    silently by the bus (C13).
    """
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    me = inner.get("agent_id")
    if bus is None or me is None:
        return _dump({"success": False, "error": "Bus or agent_id missing in context."})

    async with bus._lock:
        if target_agent_id not in bus.statuses:
            return _dump(
                {
                    "success": False,
                    "error": f"Target agent '{target_agent_id}' not found.",
                }
            )
        target_status = bus.statuses.get(target_agent_id)

    if target_status in ("completed", "crashed", "stopped"):
        return _dump(
            {
                "success": False,
                "error": f"Target agent '{target_agent_id}' is {target_status}; message dropped.",
            }
        )

    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    await bus.send(
        target_agent_id,
        {
            "id": msg_id,
            "from": me,
            "content": message,
            "type": message_type,
            "priority": priority,
        },
    )
    return _dump(
        {
            "success": True,
            "message_id": msg_id,
            "target_agent_id": target_agent_id,
            "delivery_status": "queued",
        }
    )


# Polling cadence for ``wait_for_message``. 1s matches the PLAYBOOK
# skeleton; tighter would burn CPU, slacker would feel laggy when a sibling
# delivers a message right after the wait starts.
_WAIT_POLL_SECONDS = 1.0


@strix_tool(timeout=601)
async def wait_for_message(
    ctx: RunContextWrapper,
    reason: str = "Waiting for messages from other agents",
    timeout_seconds: int = 600,
) -> str:
    """Block this agent's turn until a message arrives or ``timeout_seconds``.

    Implementation polls ``bus.inboxes`` once per second. Cheaper than an
    asyncio.Event because the message bus already serializes through its
    own lock — a missed wakeup on Event would be subtle to debug, while
    polling is trivially correct.

    Args:
        reason: Human-readable note shown in graph snapshots while waiting.
        timeout_seconds: Cap on the wait. 600s matches the legacy default.
    """
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    me = inner.get("agent_id")
    if bus is None or me is None:
        return _dump({"success": False, "error": "Bus or agent_id missing in context."})

    async with bus._lock:
        bus.statuses[me] = "waiting"

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    try:
        while asyncio.get_event_loop().time() < deadline:
            async with bus._lock:
                pending = len(bus.inboxes.get(me, []))
            if pending > 0:
                async with bus._lock:
                    bus.statuses[me] = "running"
                return _dump(
                    {
                        "success": True,
                        "status": "message_arrived",
                        "pending_messages": pending,
                        "reason": reason,
                    }
                )
            await asyncio.sleep(_WAIT_POLL_SECONDS)
    finally:
        async with bus._lock:
            # Don't clobber a status another writer set (e.g., on_agent_end
            # finalized us as ``stopped`` mid-wait).
            if bus.statuses.get(me) == "waiting":
                bus.statuses[me] = "running"

    return _dump(
        {
            "success": True,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "reason": reason,
            "note": "No messages within timeout — continue work or call agent_finish.",
        }
    )


@strix_tool(timeout=120)
async def create_agent(
    ctx: RunContextWrapper,
    name: str,
    task: str,
    inherit_context: bool = True,
    skills: list[str] | None = None,
) -> str:
    """Spawn a child agent that runs in parallel via ``asyncio.create_task``.

    The child's ``Runner.run`` task handle is stored in ``bus.tasks[child_id]``
    so a root-level cancel can cascade to descendants (C9). The child is
    registered with the bus before the task starts so messages aimed at it
    don't get dropped during the brief register→start window.

    Args:
        name: Human-readable child name (also stored in ``bus.names``).
        task: The task description handed to the child agent.
        inherit_context: When True, the child receives a copy of the parent's
            input items as background context, wrapped in
            ``<inherited_context_from_parent>``. Default True.
        skills: Optional list of skill names the child should preload.

    Returns a JSON-encoded ``{"success": ..., "agent_id": ...}``.
    """
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    parent_id = inner.get("agent_id")
    factory: Callable[..., SDKAgent] | None = inner.get("agent_factory")

    if bus is None or parent_id is None:
        return _dump({"success": False, "error": "Bus or agent_id missing in context."})
    if factory is None:
        return _dump(
            {
                "success": False,
                "error": (
                    "No agent_factory in context. "
                    "The root assembly must inject one via make_agent_context."
                ),
            }
        )

    child_id = uuid.uuid4().hex[:8]

    try:
        child_agent = factory(name=name, skills=skills or [])
    except Exception as e:
        logger.exception("agent_factory raised while building child '%s'", name)
        return _dump(
            {
                "success": False,
                "error": f"agent_factory failed: {e!s}",
            }
        )

    await bus.register(child_id, name, parent_id)

    # Build the child's input. Identity injection mirrors the legacy
    # <agent_delegation> envelope so the child's system prompt's existing
    # rules around self-identity still apply.
    parent_history = inner.get("parent_input_items") if inherit_context else None
    initial_input: list[TResponseInputItem] = []
    if parent_history:
        initial_input.append(
            {
                "role": "user",
                "content": "<inherited_context_from_parent>",
            }
        )
        initial_input.extend(parent_history)
        initial_input.append(
            {
                "role": "user",
                "content": "</inherited_context_from_parent>",
            }
        )
    initial_input.append(
        {
            "role": "user",
            "content": (
                f"<agent_delegation>\n"
                f"You are agent {name} ({child_id}). Parent is {parent_id}.\n"
                f"Maintain self-identity. Use agent_finish when complete.\n"
                f"</agent_delegation>"
            ),
        }
    )
    initial_input.append({"role": "user", "content": task})

    child_ctx = make_agent_context(
        bus=bus,
        sandbox_session=inner.get("sandbox_session"),
        sandbox_client=inner.get("sandbox_client"),
        sandbox_token=inner.get("sandbox_token"),
        tool_server_host_port=inner.get("tool_server_host_port"),
        caido_host_port=inner.get("caido_host_port"),
        agent_id=child_id,
        agent_name=name,
        parent_id=parent_id,
        tracer=inner.get("tracer"),
        model=inner.get("model", "strix/claude-sonnet-4.6"),
        model_settings=inner.get("model_settings"),
        max_turns=int(inner.get("max_turns", 300)),
        is_whitebox=bool(inner.get("is_whitebox", False)),
        diff_scope=inner.get("diff_scope"),
        run_id=inner.get("run_id"),
        agent_factory=factory,
    )

    child_run_config = make_run_config(
        sandbox_session=inner.get("sandbox_session"),
        sandbox_client=inner.get("sandbox_client"),
        model=inner.get("model", "strix/claude-sonnet-4.6"),
        model_settings_override=inner.get("model_settings"),
    )

    task_handle = asyncio.create_task(
        Runner.run(
            child_agent,
            input=initial_input,
            run_config=child_run_config,
            context=child_ctx,
            hooks=StrixOrchestrationHooks(),
            max_turns=int(inner.get("max_turns", 300)),
        ),
        name=f"agent-{name}-{child_id}",
    )
    async with bus._lock:
        bus.tasks[child_id] = task_handle

    return _dump(
        {
            "success": True,
            "agent_id": child_id,
            "name": name,
            "parent_id": parent_id,
            "message": f"Spawned '{name}' ({child_id}) running in parallel.",
        }
    )


@strix_tool(timeout=30)
async def agent_finish(
    ctx: RunContextWrapper,
    result_summary: str,
    findings: list[str] | None = None,
    success: bool = True,
    report_to_parent: bool = True,
    final_recommendations: list[str] | None = None,
) -> str:
    """Subagent-only termination: post a completion report and signal the SDK.

    Sets ``ctx.context['agent_finish_called'] = True`` so the on_agent_end
    hook records "completed" rather than "crashed". The SDK terminates the
    child's loop because every child is built with
    ``tool_use_behavior={"stop_at_tool_names": ["agent_finish"]}`` (C4).

    Root agents must call ``finish_scan`` instead. This tool refuses to run
    when ``parent_id`` is None.
    """
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    bus = inner.get("bus")
    me = inner.get("agent_id")
    if bus is None or me is None:
        return _dump({"success": False, "error": "Bus or agent_id missing in context."})

    parent_id = inner.get("parent_id")
    if parent_id is None:
        return _dump(
            {
                "success": False,
                "agent_completed": False,
                "error": (
                    "agent_finish is for subagents. Root/main agents must call finish_scan instead."
                ),
                "parent_notified": False,
            }
        )

    inner["agent_finish_called"] = True

    parent_notified = False
    if report_to_parent:
        findings_xml = "\n".join(f"        <finding>{f}</finding>" for f in (findings or []))
        rec_xml = "\n".join(
            f"        <recommendation>{r}</recommendation>" for r in (final_recommendations or [])
        )
        async with bus._lock:
            agent_name = bus.names.get(me, me)
        report = (
            f"<agent_completion_report from='{agent_name}' agent_id='{me}' "
            f"success='{success}'>\n"
            f"    <summary>{result_summary}</summary>\n"
            f"    <findings>\n{findings_xml}\n    </findings>\n"
            f"    <recommendations>\n{rec_xml}\n    </recommendations>\n"
            f"</agent_completion_report>"
        )
        await bus.send(
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

    return _dump(
        {
            "success": True,
            "agent_completed": True,
            "parent_notified": parent_notified,
            "agent_id": me,
            "summary": result_summary,
            "findings_count": len(findings or []),
            "has_recommendations": bool(final_recommendations),
        }
    )
