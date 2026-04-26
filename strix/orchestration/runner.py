"""Top-level Strix scan runner.

The SDK owns model/tool execution and per-agent sessions. This module owns
Strix-specific scan setup, child-agent startup, resume, and the small wake loop
needed to keep every agent addressable after its SDK run parks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import RunConfig, Runner
from agents.exceptions import AgentsException, MaxTurnsExceeded, UserError
from agents.memory import SQLiteSession
from agents.sandbox import SandboxRunConfig
from openai import APIError

from strix.agents.factory import build_strix_agent, make_child_factory
from strix.config import load_settings
from strix.llm.multi_provider_setup import build_multi_provider
from strix.orchestration.coordinator import AgentCoordinator, Status
from strix.orchestration.utils import (
    DEFAULT_MAX_TURNS,
    build_root_task,
    build_scope_context,
    child_initial_input,
    make_model_settings,
)
from strix.runtime import session_manager
from strix.telemetry.logging import set_scan_id, setup_scan_logging


if TYPE_CHECKING:
    from agents.memory import Session
    from agents.result import RunResultBase


logger = logging.getLogger(__name__)

StreamEventSink = Callable[[str, Any], None]


def _open_agent_session(agent_id: str, path: Path) -> SQLiteSession:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(session_id=agent_id, db_path=path)


async def _run_agent_loop(
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
    event_sink: StreamEventSink | None = None,
) -> RunResultBase | None:
    await coordinator.attach_runtime(
        agent_id,
        session=session,
        interrupt_on_message=interactive,
    )
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
            event_sink=event_sink,
        )

    if not interactive:
        return result

    while True:
        try:
            await coordinator.wait_for_message(agent_id)
        except asyncio.CancelledError:
            return result

        await coordinator.consume_pending(agent_id)
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
            event_sink=event_sink,
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
    event_sink: StreamEventSink | None,
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
                if event_sink is not None:
                    try:
                        event_sink(agent_id, event)
                    except Exception:
                        logger.exception("stream event sink failed for %s", agent_id)
            if stream.run_loop_exception is not None:
                raise stream.run_loop_exception
        finally:
            await coordinator.detach_stream(agent_id, stream)
    except Exception as exc:
        if not interactive:
            raise
        if isinstance(exc, MaxTurnsExceeded):
            status: Status = "stopped"
        elif isinstance(exc, UserError | AgentsException | APIError):
            status = "failed"
        else:
            status = "crashed"
        logger.exception("agent run failed for %s; parking as %s", agent_id, status)
        await coordinator.set_status(agent_id, status)
        await _notify_parent_on_crash(coordinator, agent_id, status)
        return None
    else:
        await _settle_run_result(coordinator, agent_id, interactive)
        return stream


async def _settle_run_result(
    coordinator: AgentCoordinator,
    agent_id: str,
    interactive: bool,
) -> None:
    async with coordinator._lock:
        current_status = coordinator.statuses.get(agent_id)

    if current_status != "running":
        return

    status: Status = "waiting" if interactive else "crashed"
    await coordinator.set_status(agent_id, status)
    await _notify_parent_on_crash(coordinator, agent_id, status)


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


async def _spawn_child_agent(
    *,
    coordinator: AgentCoordinator,
    factory: Any,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    parent_ctx: dict[str, Any],
    name: str,
    task: str,
    skills: list[str],
    parent_history: list[Any],
    event_sink: StreamEventSink | None = None,
) -> dict[str, Any]:
    parent_id = parent_ctx.get("agent_id")
    if not isinstance(parent_id, str):
        raise TypeError("Parent agent_id missing from context")

    child_id = uuid.uuid4().hex[:8]
    child_agent = factory(name=name, skills=skills)
    await coordinator.register(
        child_id,
        name,
        parent_id,
        task=task,
        skills=skills,
    )

    await _start_child_runner(
        parent_ctx=parent_ctx,
        coordinator=coordinator,
        agents_db_path=agents_db_path,
        sessions_to_close=sessions_to_close,
        run_config=run_config,
        max_turns=max_turns,
        interactive=interactive,
        child_agent=child_agent,
        child_id=child_id,
        name=name,
        parent_id=parent_id,
        task=task,
        initial_input=child_initial_input(
            name=name,
            child_id=child_id,
            parent_id=parent_id,
            task=task,
            parent_history=parent_history,
        ),
        event_sink=event_sink,
    )

    return {
        "success": True,
        "agent_id": child_id,
        "name": name,
        "parent_id": parent_id,
        "message": f"Spawned '{name}' ({child_id}) running in parallel.",
    }


async def _start_child_runner(
    *,
    parent_ctx: dict[str, Any],
    coordinator: AgentCoordinator,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    child_agent: Any,
    child_id: str,
    name: str,
    parent_id: str | None,
    task: str,
    initial_input: Any,
    start_parked: bool = False,
    event_sink: StreamEventSink | None = None,
) -> None:
    session = _open_agent_session(child_id, agents_db_path)
    sessions_to_close.append(session)
    await coordinator.attach_runtime(child_id, session=session)

    child_ctx: dict[str, Any] = dict(parent_ctx)
    child_ctx["agent_id"] = child_id
    child_ctx["parent_id"] = parent_id
    child_ctx["task"] = task

    task_handle = asyncio.create_task(
        _run_agent_loop(
            agent=child_agent,
            initial_input=initial_input,
            run_config=run_config,
            context=child_ctx,
            max_turns=max_turns,
            coordinator=coordinator,
            agent_id=child_id,
            interactive=interactive,
            session=session,
            start_parked=start_parked,
            event_sink=event_sink,
        ),
        name=f"agent-{name}-{child_id}",
    )
    await coordinator.attach_runtime(child_id, task=task_handle)


async def run_strix_scan(
    *,
    scan_config: dict[str, Any],
    scan_id: str | None = None,
    image: str,
    local_sources: list[dict[str, str]] | None = None,
    coordinator: AgentCoordinator | None = None,
    interactive: bool = False,
    max_turns: int = DEFAULT_MAX_TURNS,
    model: str | None = None,
    cleanup_on_exit: bool = True,
    event_sink: StreamEventSink | None = None,
) -> RunResultBase | None:
    """Run or resume one Strix scan against a sandbox."""
    if scan_id is None:
        scan_id = f"scan-{uuid.uuid4().hex[:8]}"

    # Resolve run_dir before any heavy bring-up so the log file captures
    # everything from sandbox start onwards.
    run_dir = Path.cwd() / "strix_runs" / scan_id
    run_dir.mkdir(parents=True, exist_ok=True)
    teardown_logging = setup_scan_logging(run_dir)
    set_scan_id(scan_id)

    agents_path = run_dir / "agents.json"
    agents_db = run_dir / "agents.db"
    is_resume = agents_path.exists()

    logger.info(
        "%s Strix scan %s (image=%s, max_turns=%d, interactive=%s, run_dir=%s)",
        "Resuming" if is_resume else "Starting",
        scan_id,
        image,
        max_turns,
        interactive,
        run_dir,
    )

    settings = load_settings()
    resolved_model = model or settings.llm.model
    if not resolved_model:
        raise RuntimeError(
            "No LLM model configured. Set STRIX_LLM env or pass model= to run_strix_scan().",
        )
    logger.info("LLM model resolved: %s", resolved_model)

    # Caller may pre-create the coordinator so it can route stop/chat
    # commands while the scan loop runs in another thread.
    if coordinator is None:
        coordinator = AgentCoordinator()
    coordinator.set_snapshot_path(agents_path)

    # Wire the per-agent todo store to ``{run_dir}/todos.json`` (mirrored
    # on every CRUD) and reload any prior todos so respawned subagents
    # find their lists intact. Same for the shared notes store.
    from strix.tools.notes.tools import hydrate_notes_from_disk
    from strix.tools.todo.tools import hydrate_todos_from_disk

    hydrate_todos_from_disk(run_dir)
    hydrate_notes_from_disk(run_dir)

    root_id: str | None = None
    if is_resume:
        try:
            snap = json.loads(agents_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot resume scan {scan_id}: agents.json is unreadable: {exc}",
            ) from exc
        if not agents_db.exists():
            raise RuntimeError(
                f"Cannot resume scan {scan_id}: missing SDK session database at {agents_db}",
            )
        await coordinator.restore(snap)
        for aid, parent in coordinator.parent_of.items():
            if parent is None:
                root_id = aid
                break
        if root_id is None:
            raise RuntimeError(
                f"Cannot resume scan {scan_id}: agents.json has no root agent (parent=None)",
            )
        logger.info(
            "Resume: restored coordinator with %d agent(s); root=%s",
            len(coordinator.statuses),
            root_id,
        )
    else:
        root_id = uuid.uuid4().hex[:8]

    logger.info("Bringing up sandbox session for scan %s", scan_id)
    bundle = await session_manager.create_or_reuse(
        scan_id,
        image=image,
        local_sources=local_sources or [],
    )
    logger.info("Sandbox ready for scan %s", scan_id)

    sessions_to_close: list[SQLiteSession] = []

    try:
        targets = scan_config.get("targets") or []
        scan_mode = str(scan_config.get("scan_mode") or "deep")
        is_whitebox = any(t.get("type") == "local_code" for t in targets)
        skills = list(scan_config.get("skills") or [])
        root_task = build_root_task(scan_config)
        model_settings = make_model_settings(settings.llm.reasoning_effort)
        run_config = RunConfig(
            model=resolved_model,
            model_provider=build_multi_provider(),
            model_settings=model_settings,
            sandbox=SandboxRunConfig(client=bundle["client"], session=bundle["session"]),
            trace_include_sensitive_data=False,
        )

        scope_context = build_scope_context(scan_config)

        root_agent = build_strix_agent(
            name="strix",
            skills=skills,
            is_root=True,
            scan_mode=scan_mode,
            is_whitebox=is_whitebox,
            interactive=interactive,
            system_prompt_context=scope_context,
        )

        if not is_resume:
            await coordinator.register(
                root_id,
                "strix",
                parent_id=None,
                task=root_task,
                skills=skills,
            )

        child_agent_builder = make_child_factory(
            scan_mode=scan_mode,
            is_whitebox=is_whitebox,
            interactive=interactive,
            system_prompt_context=scope_context,
        )

        async def spawn_child_agent(**kwargs: Any) -> dict[str, Any]:
            return await _spawn_child_agent(
                coordinator=coordinator,
                factory=child_agent_builder,
                agents_db_path=agents_db,
                sessions_to_close=sessions_to_close,
                run_config=run_config,
                max_turns=max_turns,
                interactive=interactive,
                event_sink=event_sink,
                **kwargs,
            )

        context: dict[str, Any] = {
            "coordinator": coordinator,
            "sandbox_session": bundle["session"],
            "caido_client": bundle["caido_client"],
            "agent_id": root_id,
            "parent_id": None,
            "interactive": interactive,
            "spawn_child_agent": spawn_child_agent,
        }

        # All agents share one SQLite database; SDK session_id separates
        # each agent's conversation inside that database.
        root_session = _open_agent_session(root_id, agents_db)
        sessions_to_close.append(root_session)
        await coordinator.attach_runtime(root_id, session=root_session)

        if is_resume:
            await _respawn_subagents(
                coordinator=coordinator,
                factory=child_agent_builder,
                agents_db_path=agents_db,
                sessions_to_close=sessions_to_close,
                run_config=run_config,
                max_turns=max_turns,
                interactive=interactive,
                parent_ctx=context,
                root_id=root_id,
                event_sink=event_sink,
            )

        initial_input: Any = [] if is_resume else root_task

        # Resume + new ``--instruction``: SDK replay drives root from
        # agents.db with ``initial_input=[]``, so a brand-new instruction
        # passed on the resume CLI would otherwise be silently ignored.
        # Inject it as a fresh user message in root's SDK session; the
        # next run cycle will replay it with the rest of the session.
        resume_instruction = str(scan_config.get("resume_instruction") or "").strip()
        if is_resume and resume_instruction:
            await coordinator.send(
                root_id,
                {
                    "from": "user",
                    "type": "instruction",
                    "priority": "high",
                    "content": resume_instruction,
                },
            )
            logger.info(
                "Resume: injected new instruction into root SDK session (len=%d)",
                len(resume_instruction),
            )

        async with coordinator._lock:
            root_status = coordinator.statuses.get(root_id)

        return await _run_agent_loop(
            agent=root_agent,
            initial_input=initial_input,
            run_config=run_config,
            context=context,
            max_turns=max_turns,
            coordinator=coordinator,
            agent_id=root_id,
            interactive=interactive,
            session=root_session,
            start_parked=bool(interactive and is_resume and root_status != "running"),
            event_sink=event_sink,
        )
    except BaseException:
        logger.exception("Strix scan %s failed", scan_id)
        # Cancel any descendant tasks the root spawned before unwinding.
        # cancel_descendants is idempotent and handles the empty-tree case.
        if root_id is not None:
            await coordinator.cancel_descendants(root_id)
            # The SDK's on_agent_end hook only fires after a successful
            # ``Runner.run_streamed`` reaches the agent's first turn. A
            # failure earlier (e.g., model-provider routing, sandbox
            # bring-up) leaves the root stuck at status="running" — the
            # TUI keeps animating "Initializing" forever. Finalize it
            # here so the coordinator reflects reality.
            with contextlib.suppress(Exception):
                await coordinator.set_status(root_id, "failed")
        raise
    finally:
        for s in sessions_to_close:
            with contextlib.suppress(Exception):
                s.close()
        with contextlib.suppress(Exception):
            await coordinator._maybe_snapshot()
        if cleanup_on_exit:
            logger.info("Tearing down sandbox session for scan %s", scan_id)
            await session_manager.cleanup(scan_id)
        logger.info("Strix scan %s done", scan_id)
        teardown_logging()


async def _respawn_subagents(
    *,
    coordinator: AgentCoordinator,
    factory: Any,
    agents_db_path: Path,
    sessions_to_close: list[SQLiteSession],
    run_config: RunConfig,
    max_turns: int,
    interactive: bool,
    parent_ctx: dict[str, Any],
    root_id: str,
    event_sink: StreamEventSink | None = None,
) -> None:
    """Re-spawn subagent runners from a restored coordinator snapshot.

    Each child gets its own SDK ``session_id`` inside the shared
    ``agents.db`` so the SDK replays its prior conversation. Interactive
    mode respawns every registered child as a parked runner unless it was
    actively running before the crash. That keeps completed/stopped/
    crashed/failed agents addressable: a later message wakes the SDK
    session instead of being dropped into a dead inbox.
    """
    async with coordinator._lock:
        # Snapshot the iteration view first so we can mutate via coordinator
        # below without "dict changed during iteration" trouble.
        agents_snapshot = [
            (aid, status, dict(coordinator.metadata.get(aid, {})))
            for aid, status in coordinator.statuses.items()
        ]
        candidates: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for aid, status, md in agents_snapshot:
            if not interactive and status not in {"running", "waiting"}:
                continue
            if coordinator.parent_of.get(aid) is None or aid == root_id:
                continue
            md["_restored_status"] = status
            candidates.append(
                (
                    aid,
                    coordinator.names.get(aid, aid),
                    coordinator.parent_of.get(aid),
                    md,
                )
            )

    for child_id, name, parent_id, md in candidates:
        try:
            restored_status = str(md.get("_restored_status") or "running")
            start_parked = interactive and restored_status != "running"

            if start_parked:
                logger.warning(
                    "respawn %s (%s): starting parked from status=%s",
                    child_id,
                    name,
                    restored_status,
                )

            child_skills = list(md.get("skills") or [])
            child_agent = factory(name=name, skills=child_skills)
            await _start_child_runner(
                parent_ctx=parent_ctx,
                coordinator=coordinator,
                agents_db_path=agents_db_path,
                sessions_to_close=sessions_to_close,
                run_config=run_config,
                max_turns=max_turns,
                interactive=interactive,
                child_agent=child_agent,
                child_id=child_id,
                name=name,
                parent_id=parent_id,
                task=str(md.get("task", "")),
                initial_input=[],
                start_parked=start_parked,
                event_sink=event_sink,
            )
            logger.info(
                "respawned %s (%s) parent=%s task_len=%d",
                child_id,
                name,
                parent_id or "-",
                len(md.get("task", "")),
            )
        except Exception:
            logger.exception("respawn %s failed; marking crashed", child_id)
            with contextlib.suppress(Exception):
                await coordinator.set_status(child_id, "crashed")
