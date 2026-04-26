"""SDK-native coordinator for Strix's addressable agent graph.

The Agents SDK owns model/tool execution and per-agent conversation
history. Strix owns only product semantics the SDK does not provide:
agent ids, the parent/child graph, wake/stop signals, TUI-visible
status, and process-resume metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agents import Runner
from agents.exceptions import AgentsException, MaxTurnsExceeded, UserError
from agents.memory import SQLiteSession
from openai import APIError


if TYPE_CHECKING:
    from agents.items import TResponseInputItem
    from agents.memory import Session
    from agents.result import RunResultBase, RunResultStreaming
    from agents.run_config import RunConfig


logger = logging.getLogger(__name__)

Status = Literal["running", "waiting", "completed", "stopped", "crashed", "failed", "llm_failed"]
ACTIVE_AGENT_STATUSES = {"running", "waiting", "llm_failed"}
_SNAPSHOT_VERSION = 3
_WAITING_TIMEOUT_SUBAGENT = 300.0
_TIMEOUT_RESUME_MESSAGE = "Waiting timeout reached. Resuming execution."


@dataclass(slots=True)
class AgentRuntime:
    session: Session | None = None
    task: asyncio.Task[Any] | None = None
    stream: RunResultStreaming | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    tool_calls: dict[str, str] = field(default_factory=dict)


class AgentCoordinator:
    """Single owner for graph state, SDK runtimes, messages, and resume snapshots."""

    def __init__(self) -> None:
        self.statuses: dict[str, Status] = {}
        self.parent_of: dict[str, str | None] = {}
        self.names: dict[str, str] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.pending_counts: dict[str, int] = {}
        self.pending_user_counts: dict[str, int] = {}
        self.queued_messages: dict[str, list[dict[str, Any]]] = {}
        self.stats_live: dict[str, dict[str, Any]] = {}
        self.runtimes: dict[str, AgentRuntime] = {}
        self._lock = asyncio.Lock()
        self._snapshot_path: Path | None = None

    def set_snapshot_path(self, path: Path) -> None:
        self._snapshot_path = path

    async def register(
        self,
        agent_id: str,
        name: str,
        parent_id: str | None,
        *,
        task: str | None = None,
        skills: list[str] | None = None,
        is_whitebox: bool = False,
        scan_mode: str = "deep",
        diff_scope: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self.statuses[agent_id] = "running"
            self.parent_of[agent_id] = parent_id
            self.names[agent_id] = name
            self.pending_counts.setdefault(agent_id, 0)
            self.pending_user_counts.setdefault(agent_id, 0)
            self.stats_live.setdefault(agent_id, _empty_stats())
            self.metadata[agent_id] = {
                "task": task or "",
                "skills": list(skills or []),
                "is_whitebox": bool(is_whitebox),
                "scan_mode": scan_mode,
                "diff_scope": diff_scope,
            }
            self.runtimes.setdefault(agent_id, AgentRuntime())
        logger.info("agent.register %s (%s) parent=%s", agent_id, name, parent_id or "-")
        await self._maybe_snapshot()

    async def attach_runtime(
        self,
        agent_id: str,
        *,
        session: Session | None = None,
        task: asyncio.Task[Any] | None = None,
    ) -> None:
        async with self._lock:
            runtime = self.runtimes.setdefault(agent_id, AgentRuntime())
            if session is not None:
                runtime.session = session
            if task is not None:
                runtime.task = task
        if session is not None:
            await self.flush_queued_messages(agent_id)

    async def mark_running(self, agent_id: str) -> None:
        async with self._lock:
            if agent_id in self.statuses:
                self.statuses[agent_id] = "running"
        await self._maybe_snapshot()

    async def park_waiting(self, agent_id: str) -> None:
        await self.set_status(agent_id, "waiting")

    async def mark_llm_failed(self, agent_id: str) -> None:
        await self.set_status(agent_id, "llm_failed")

    async def set_status(self, agent_id: str, status: Status | str) -> None:
        async with self._lock:
            if agent_id not in self.statuses:
                return
            self.statuses[agent_id] = status  # type: ignore[assignment]
            runtime = self.runtimes.setdefault(agent_id, AgentRuntime())
            runtime.wake.set()
        logger.info("agent.status %s=%s", agent_id, status)
        await self._maybe_snapshot()

    async def send(self, target_agent_id: str, message: dict[str, Any]) -> bool:
        """Deliver a user/peer message by appending it to the target SDK session."""
        should_queue = False
        async with self._lock:
            if target_agent_id not in self.statuses:
                logger.debug("agent.send dropped unknown target=%s", target_agent_id)
                return False
            runtime = self.runtimes.setdefault(target_agent_id, AgentRuntime())
            session = runtime.session
            should_queue = session is None or runtime.stream is not None
            if should_queue:
                self.queued_messages.setdefault(target_agent_id, []).append(dict(message))
        if session is not None and not should_queue:
            try:
                await session.add_items([self._message_to_session_item(message)])
            except Exception:
                logger.exception(
                    "agent.send failed to append to SDK session target=%s",
                    target_agent_id,
                )
                return False
        async with self._lock:
            self.pending_counts[target_agent_id] = self.pending_counts.get(target_agent_id, 0) + 1
            if message.get("from") == "user":
                self.pending_user_counts[target_agent_id] = (
                    self.pending_user_counts.get(target_agent_id, 0) + 1
                )
            self.runtimes.setdefault(target_agent_id, AgentRuntime()).wake.set()
        if should_queue:
            logger.debug(
                "agent.send %s queued until SDK session is safe to append", target_agent_id
            )
        await self._maybe_snapshot()
        return True

    async def flush_queued_messages(self, agent_id: str) -> None:
        async with self._lock:
            runtime = self.runtimes.setdefault(agent_id, AgentRuntime())
            session = runtime.session
            queued = self.queued_messages.pop(agent_id, [])
        if not queued:
            return
        if session is None:
            async with self._lock:
                self.queued_messages.setdefault(agent_id, []).extend(queued)
            return
        try:
            await session.add_items([self._message_to_session_item(msg) for msg in queued])
        except Exception:
            async with self._lock:
                self.queued_messages.setdefault(agent_id, [])[0:0] = queued
            logger.exception("agent.flush_queued_messages failed for %s", agent_id)

    async def wait_for_message(self, agent_id: str, *, user_only: bool = False) -> None:
        while True:
            async with self._lock:
                pending = (
                    self.pending_user_counts.get(agent_id, 0)
                    if user_only
                    else self.pending_counts.get(agent_id, 0)
                )
                if pending > 0:
                    return
                wake = self.runtimes.setdefault(agent_id, AgentRuntime()).wake
                wake.clear()
            await wake.wait()

    async def wait_for_user_message(self, agent_id: str) -> None:
        await self.wait_for_message(agent_id, user_only=True)

    async def consume_wake(self, agent_id: str) -> None:
        async with self._lock:
            self.pending_counts[agent_id] = 0
            self.pending_user_counts[agent_id] = 0

    async def pending_count(self, agent_id: str) -> int:
        async with self._lock:
            return self.pending_counts.get(agent_id, 0)

    async def recent_session_items(self, agent_id: str, count: int) -> list[TResponseInputItem]:
        if count <= 0:
            return []
        async with self._lock:
            session = self.runtimes.get(agent_id, AgentRuntime()).session
        if session is None:
            return []
        items = await session.get_items()
        return list(items[-count:])

    async def request_interrupt(self, agent_id: str, mode: str = "after_turn") -> bool:
        async with self._lock:
            stream = self.runtimes.get(agent_id, AgentRuntime()).stream
        if stream is None:
            return False
        stream.cancel(mode=mode)  # type: ignore[arg-type]
        return True

    async def request_stop(self, agent_id: str, *, interrupt: bool = True) -> None:
        async with self._lock:
            if agent_id not in self.statuses:
                return
            self.statuses[agent_id] = "stopped"
            runtime = self.runtimes.setdefault(agent_id, AgentRuntime())
            runtime.wake.set()
            stream = runtime.stream
        if interrupt and stream is not None:
            stream.cancel(mode="after_turn")
        await self._maybe_snapshot()

    async def cancel_descendants(self, agent_id: str) -> None:
        tasks = []
        async with self._lock:
            for aid in reversed(self._subtree_order_locked(agent_id)):
                task = self.runtimes.get(aid, AgentRuntime()).task
                if task is not None and not task.done():
                    tasks.append(task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_descendants_graceful(self, agent_id: str) -> None:
        async with self._lock:
            order = self._subtree_order_locked(agent_id)
        for aid in reversed(order):
            await self.request_stop(aid)
        await self._maybe_snapshot()

    async def attach_stream(
        self,
        agent_id: str,
        stream: RunResultStreaming,
    ) -> None:
        async with self._lock:
            self.runtimes.setdefault(agent_id, AgentRuntime()).stream = stream

    async def detach_stream(
        self,
        agent_id: str,
        stream: RunResultStreaming,
    ) -> None:
        async with self._lock:
            runtime = self.runtimes.setdefault(agent_id, AgentRuntime())
            if runtime.stream is stream:
                runtime.stream = None

    async def active_agents_except(self, agent_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                self._agent_info_locked(aid)
                for aid, status in self.statuses.items()
                if aid != agent_id and status in ACTIVE_AGENT_STATUSES
            ]

    async def agent_info(self, agent_id: str) -> dict[str, Any] | None:
        async with self._lock:
            if agent_id not in self.statuses:
                return None
            return self._agent_info_locked(agent_id)

    async def graph_snapshot(
        self,
    ) -> tuple[dict[str, str | None], dict[str, Status], dict[str, str]]:
        async with self._lock:
            return dict(self.parent_of), dict(self.statuses), dict(self.names)

    async def record_usage(self, agent_id: str, usage: Any) -> None:
        if usage is None:
            return
        async with self._lock:
            stats = self.stats_live.setdefault(agent_id, _empty_stats())
            stats["calls"] += 1
            stats["in"] += getattr(usage, "input_tokens", 0) or 0
            stats["out"] += getattr(usage, "output_tokens", 0) or 0
            details = getattr(usage, "input_tokens_details", None)
            stats["cached"] += getattr(details, "cached_tokens", 0) or 0 if details else 0

    def _message_to_session_item(self, message: dict[str, Any]) -> TResponseInputItem:
        sender = str(message.get("from", "unknown"))
        content = str(message.get("content", ""))
        if sender == "user":
            return {"role": "user", "content": content}
        sender_name = self.names.get(sender, sender)
        msg_type = message.get("type", "information")
        priority = message.get("priority", "normal")
        return {
            "role": "user",
            "content": (
                f"[Message from {sender_name} ({sender}) | type={msg_type} "
                f"| priority={priority}]\n{content}"
            ),
        }

    def _subtree_order_locked(self, agent_id: str) -> list[str]:
        queue = [agent_id]
        order: list[str] = []
        while queue:
            aid = queue.pop()
            order.append(aid)
            queue.extend(child for child, parent in self.parent_of.items() if parent == aid)
        return order

    def _agent_info_locked(self, agent_id: str) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "name": self.names.get(agent_id, agent_id),
            "status": self.statuses.get(agent_id),
            "parent_id": self.parent_of.get(agent_id),
            "pending_messages": self.pending_counts.get(agent_id, 0),
        }

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "version": _SNAPSHOT_VERSION,
                "statuses": dict(self.statuses),
                "parent_of": dict(self.parent_of),
                "names": dict(self.names),
                "metadata": {aid: dict(md) for aid, md in self.metadata.items()},
                "pending_counts": dict(self.pending_counts),
                "pending_user_counts": dict(self.pending_user_counts),
                "queued_messages": {
                    aid: [dict(msg) for msg in msgs] for aid, msgs in self.queued_messages.items()
                },
                "stats_live": {aid: dict(s) for aid, s in self.stats_live.items()},
            }

    async def restore(self, snap: dict[str, Any]) -> None:
        async with self._lock:
            self.statuses = dict(snap.get("statuses", {}))
            self.parent_of = dict(snap.get("parent_of", {}))
            self.names = dict(snap.get("names", {}))
            self.metadata = {aid: dict(md) for aid, md in snap.get("metadata", {}).items()}
            self.pending_counts = dict(snap.get("pending_counts", {}))
            self.pending_user_counts = dict(snap.get("pending_user_counts", {}))
            self.queued_messages = {
                aid: [dict(msg) for msg in msgs]
                for aid, msgs in snap.get("queued_messages", {}).items()
            }
            self.stats_live = {aid: dict(s) for aid, s in snap.get("stats_live", {}).items()}
            for aid in self.statuses:
                self.runtimes.setdefault(aid, AgentRuntime())

    async def _maybe_snapshot(self) -> None:
        path = self._snapshot_path
        if path is None:
            return
        try:
            data = await self.snapshot()
            payload = json.dumps(data, ensure_ascii=False, default=str)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        except Exception:
            logger.exception("coordinator snapshot to %s failed", path)


def coordinator_from_context(ctx: dict[str, Any]) -> AgentCoordinator | None:
    coordinator = ctx.get("coordinator")
    return coordinator if isinstance(coordinator, AgentCoordinator) else None


async def run_with_continuation(
    *,
    agent: Any,
    initial_input: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
    session: Session | None = None,
    start_parked: bool = False,
) -> RunResultBase | None:
    await coordinator.attach_runtime(agent_id, session=session)
    waiting_timeout = await _waiting_timeout(coordinator, agent_id, interactive)
    result: RunResultBase | None = None

    if not (start_parked and interactive):
        result = await _run_cycle(
            agent,
            coordinator,
            agent_id,
            input_data=initial_input,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            session=session,
            interactive=interactive,
        )

    if not interactive:
        return result

    while True:
        async with coordinator._lock:
            status = coordinator.statuses.get(agent_id)

        try:
            if status == "llm_failed":
                await coordinator.wait_for_user_message(agent_id)
            elif waiting_timeout is None:
                await coordinator.wait_for_message(agent_id)
            else:
                await asyncio.wait_for(
                    coordinator.wait_for_message(agent_id),
                    timeout=waiting_timeout,
                )
        except asyncio.CancelledError:
            return result
        except TimeoutError:
            result = await _run_cycle(
                agent,
                coordinator,
                agent_id,
                input_data=_TIMEOUT_RESUME_MESSAGE,
                run_config=run_config,
                context=context,
                max_turns=max_turns,
                session=session,
                interactive=interactive,
            )
            continue

        await coordinator.consume_wake(agent_id)
        result = await _run_cycle(
            agent,
            coordinator,
            agent_id,
            input_data=[],
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            session=session,
            interactive=interactive,
        )


async def _waiting_timeout(
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
) -> float | None:
    if not interactive:
        return None
    async with coordinator._lock:
        return (
            _WAITING_TIMEOUT_SUBAGENT if coordinator.parent_of.get(agent_id) is not None else None
        )


async def _run_cycle(
    agent: Any,
    coordinator: AgentCoordinator,
    agent_id: str,
    *,
    input_data: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    max_turns: int,
    session: Session | None,
    interactive: bool,
) -> RunResultBase | None:
    try:
        await coordinator.mark_running(agent_id)
        stream = Runner.run_streamed(
            agent,
            input=input_data,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            session=session,
        )
        await coordinator.attach_stream(agent_id, stream)
        try:
            async for event in stream.stream_events():
                await _handle_stream_event(coordinator, agent_id, context, event)
        finally:
            await coordinator.detach_stream(agent_id, stream)
            await coordinator.flush_queued_messages(agent_id)
        await _settle_run_result(coordinator, agent_id, stream.final_output, interactive, context)
        return stream
    except (AgentsException, APIError):
        if not interactive:
            raise
        logger.exception("LLM/runtime failure for %s; waiting for user resume", agent_id)
        await coordinator.mark_llm_failed(agent_id)
        return None
    except Exception as exc:
        if not interactive:
            raise
        status: Status = "stopped" if isinstance(exc, MaxTurnsExceeded) else "crashed"
        if isinstance(exc, UserError):
            status = "failed"
        logger.exception("agent run failed for %s; parking as %s", agent_id, status)
        await coordinator.set_status(agent_id, status)
        await _notify_parent_on_crash(coordinator, agent_id, status)
        _mirror_tracer_status(context, coordinator, agent_id, status)
        return None


async def _settle_run_result(
    coordinator: AgentCoordinator,
    agent_id: str,
    final_output: Any,
    interactive: bool,
    context: dict[str, Any],
) -> None:
    parsed = _parse_final_output(final_output)
    async with coordinator._lock:
        current_status = coordinator.statuses.get(agent_id)

    if current_status == "stopped":
        status: Status = "stopped"
    elif parsed.get("agent_completed") or parsed.get("scan_completed"):
        status = "completed"
    elif parsed.get("agent_waiting") or interactive:
        status = "waiting"
    else:
        status = "crashed"

    await coordinator.set_status(agent_id, status)
    await _notify_parent_on_crash(coordinator, agent_id, status)
    _mirror_tracer_status(context, coordinator, agent_id, status)


def _parse_final_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, str):
        return {}
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) and parsed.get("success") else {}


async def _notify_parent_on_crash(
    coordinator: AgentCoordinator,
    agent_id: str,
    status: str,
) -> None:
    if status != "crashed":
        return
    async with coordinator._lock:
        parent = coordinator.parent_of.get(agent_id)
        name = coordinator.names.get(agent_id, agent_id)
    if parent is None:
        return
    await coordinator.send(
        parent,
        {
            "from": agent_id,
            "type": "crash",
            "priority": "high",
            "content": (
                f"[Agent crash] {name} ({agent_id}) terminated unexpectedly. "
                "Stop waiting on this child unless you want to message it again."
            ),
        },
    )


def _mirror_tracer_status(
    context: dict[str, Any],
    coordinator: AgentCoordinator,
    agent_id: str,
    status: str,
) -> None:
    tracer = context.get("tracer")
    if tracer is None:
        return
    now = datetime.now(UTC).isoformat()
    tracer.agents.setdefault(
        agent_id,
        {
            "id": agent_id,
            "name": coordinator.names.get(agent_id, agent_id),
            "parent_id": coordinator.parent_of.get(agent_id),
            "created_at": now,
        },
    )
    tracer.agents[agent_id]["status"] = status
    tracer.agents[agent_id]["updated_at"] = now


async def _handle_stream_event(
    coordinator: AgentCoordinator,
    agent_id: str,
    context: dict[str, Any],
    event: Any,
) -> None:
    tracer = context.get("tracer")
    if event.type == "raw_response_event":
        response = getattr(event.data, "response", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            await coordinator.record_usage(agent_id, usage)
        if usage is not None and tracer is not None and hasattr(tracer, "record_llm_usage"):
            details = getattr(usage, "input_tokens_details", None)
            tracer.record_llm_usage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                cached_tokens=int(getattr(details, "cached_tokens", 0) or 0) if details else 0,
            )
        return
    if tracer is None or event.type != "run_item_stream_event":
        return
    item = event.item
    raw = getattr(item, "raw_item", None)
    if event.name == "tool_called":
        call_id = str(getattr(raw, "call_id", None) or getattr(raw, "id", ""))
        tool_name = str(getattr(raw, "name", None) or getattr(raw, "type", "tool"))
        args = _parse_tool_args(getattr(raw, "arguments", None))
        runtime = coordinator.runtimes.setdefault(agent_id, AgentRuntime())
        if call_id:
            runtime.tool_calls[call_id] = tool_name
        tracer.log_tool_start(agent_id, tool_name, args)
    elif event.name == "tool_output":
        call_id = str(getattr(raw, "call_id", None) or "")
        runtime = coordinator.runtimes.setdefault(agent_id, AgentRuntime())
        tool_name = runtime.tool_calls.get(call_id, "tool")
        tracer.log_tool_end(agent_id, tool_name, _dump_raw(raw))


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_raw(raw: Any) -> Any:
    if hasattr(raw, "model_dump"):
        return raw.model_dump(exclude_unset=True)
    return raw


def open_agent_session(agent_id: str, path: Path) -> SQLiteSession:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(session_id=agent_id, db_path=path)


def _empty_stats() -> dict[str, Any]:
    return {
        "in": 0,
        "out": 0,
        "cached": 0,
        "cost": 0.0,
        "calls": 0,
    }
