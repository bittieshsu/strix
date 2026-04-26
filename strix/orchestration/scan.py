"""Top-level scan entry point with auto-resume.

1. Build (or take from caller) the per-scan ``AgentCoordinator``.
2. Wire a snapshot path so lifecycle events auto-persist ``agents.json``.
3. Acquire an advisory file lock so a second ``strix`` process can't run
   on the same ``scan_id`` concurrently.
4. **Resume detection**: if ``{run_dir}/agents.json`` already exists, restore
   the coordinator, hydrate the tracer, reuse the persisted ``root_id`` instead
   of generating a fresh one, and respawn every non-terminal subagent
   from the shared SDK ``agents.db`` before starting the root.
5. Bring up (or reuse) a sandbox session for ``scan_id``.
6. Build the root ``Agent`` + child factory.
7. Open root ``SQLiteSession`` in ``agents.db`` so the SDK replays prior
   turns on resume.
8. Call ``Runner.run`` (via ``run_with_continuation``).
9. ``finally``: close every per-agent session, take a final snapshot,
   tear down the sandbox, release the lock.

Resume is **always on**: there is no flag — presence of ``agents.json`` is
the trigger. Fresh runs simply have no ``agents.json`` to begin with.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agents import RunConfig
from agents.memory import SQLiteSession
from agents.model_settings import ModelSettings
from agents.sandbox import SandboxRunConfig
from openai.types.shared import Reasoning

from strix.agents.factory import build_strix_agent, make_child_factory
from strix.config import load_settings
from strix.llm.multi_provider_setup import build_multi_provider
from strix.llm.retry import DEFAULT_RETRY
from strix.orchestration.coordinator import (
    AgentCoordinator,
    open_agent_session,
    run_with_continuation,
)
from strix.runtime import session_manager
from strix.telemetry.logging import set_agent_id, set_scan_id, setup_scan_logging


#: Default ``max_turns`` budget passed to ``Runner.run``.
_MAX_TURNS = 300


if TYPE_CHECKING:
    from agents.result import RunResultBase


logger = logging.getLogger(__name__)


def _build_root_task(scan_config: dict[str, Any]) -> str:
    """Format the user-facing task for the root agent.

    Collects each target type into a labelled section, appends
    diff-scope context if active, and tacks on user_instructions. The
    structured section headers are referenced by the system prompt
    template, so the shape matters for prompt parity.
    """
    targets = scan_config.get("targets", []) or []
    diff_scope = scan_config.get("diff_scope") or {}
    user_instructions = scan_config.get("user_instructions", "") or ""

    repos: list[str] = []
    locals_: list[str] = []
    urls: list[str] = []
    ips: list[str] = []

    for target in targets:
        ttype = target.get("type")
        details = target.get("details") or {}
        workspace_subdir = details.get("workspace_subdir")
        workspace_path = f"/workspace/{workspace_subdir}" if workspace_subdir else "/workspace"

        if ttype == "repository":
            url = details.get("target_repo", "")
            cloned = details.get("cloned_repo_path")
            repos.append(
                f"- {url} (available at: {workspace_path})" if cloned else f"- {url}",
            )
        elif ttype == "local_code":
            path = details.get("target_path", "unknown")
            locals_.append(f"- {path} (available at: {workspace_path})")
        elif ttype == "web_application":
            urls.append(f"- {details.get('target_url', '')}")
        elif ttype == "ip_address":
            ips.append(f"- {details.get('target_ip', '')}")

    parts: list[str] = []
    if repos:
        parts.append("\n\nRepositories:")
        parts.extend(repos)
    if locals_:
        parts.append("\n\nLocal Codebases:")
        parts.extend(locals_)
    if urls:
        parts.append("\n\nURLs:")
        parts.extend(urls)
    if ips:
        parts.append("\n\nIP Addresses:")
        parts.extend(ips)

    if diff_scope.get("active"):
        parts.append("\n\nScope Constraints:")
        parts.append(
            "- Pull request diff-scope mode is active. Prioritize changed files "
            "and use other files only for context.",
        )
        for repo_scope in diff_scope.get("repos", []) or []:
            label = (
                repo_scope.get("workspace_subdir") or repo_scope.get("source_path") or "repository"
            )
            changed = repo_scope.get("analyzable_files_count", 0)
            deleted = repo_scope.get("deleted_files_count", 0)
            parts.append(f"- {label}: {changed} changed file(s) in primary scope")
            if deleted:
                parts.append(f"- {label}: {deleted} deleted file(s) are context-only")

    task = " ".join(parts)
    if user_instructions:
        task = f"{task}\n\nSpecial instructions: {user_instructions}"
    return task


def _build_scope_context(scan_config: dict[str, Any]) -> dict[str, Any]:
    """Produce the system_prompt_context block used by the prompt template.

    The prompt template's ``system_prompt_context.authorized_targets``
    lookups expect this exact shape.
    """
    authorized: list[dict[str, str]] = []
    for target in scan_config.get("targets", []) or []:
        ttype = target.get("type", "unknown")
        details = target.get("details") or {}

        if ttype == "repository":
            value = details.get("target_repo", "")
        elif ttype == "local_code":
            value = details.get("target_path", "")
        elif ttype == "web_application":
            value = details.get("target_url", "")
        elif ttype == "ip_address":
            value = details.get("target_ip", "")
        else:
            value = target.get("original", "")

        workspace_subdir = details.get("workspace_subdir")
        workspace_path = f"/workspace/{workspace_subdir}" if workspace_subdir else ""
        authorized.append(
            {"type": ttype, "value": value, "workspace_path": workspace_path},
        )

    return {
        "scope_source": "system_scan_config",
        "authorization_source": "strix_platform_verified_targets",
        "authorized_targets": authorized,
        "user_instructions_do_not_expand_scope": True,
    }


async def run_strix_scan(
    *,
    scan_config: dict[str, Any],
    scan_id: str | None = None,
    image: str,
    local_sources: list[dict[str, str]] | None = None,
    tracer: Any | None = None,
    coordinator: AgentCoordinator | None = None,
    interactive: bool = False,
    max_turns: int = _MAX_TURNS,
    model: str | None = None,
    cleanup_on_exit: bool = True,
) -> RunResultBase | None:
    """Run one Strix scan end-to-end against a freshly-prepared sandbox.

    Args:
        scan_config: Per-scan configuration — ``targets``,
            ``user_instructions``, ``diff_scope``, ``scan_mode``,
            ``skills``. ``is_whitebox`` is derived from ``targets``.
        scan_id: Used to key the sandbox session cache. Auto-generated
            if omitted — callers that want resume-after-crash semantics
            should pass a stable id.
        image: Docker image tag for the sandbox (e.g.
            ``"strix-sandbox:0.2.0"``).
        local_sources: Per-source mount specs from
            :func:`strix.interface.utils.collect_local_sources` —
            each entry's ``source_path`` (host) is bind-mounted at
            ``/workspace/<workspace_subdir>``. Pass ``None`` (or ``[]``)
            for non-whitebox runs.
        tracer: Optional Strix tracer. Stored in context for the
            telemetry hook chain. Pass ``None`` for unit tests.
        interactive: Renders the interactive-mode prompt block on the
            root agent.
        max_turns: Cap on root-agent LLM turns (default 300).
        model: Litellm model alias. ``None`` (default) reads
            :attr:`Settings.llm.model` — caller pre-validates via
            :func:`validate_environment` that it's set.
        cleanup_on_exit: When True (default), tears down the sandbox
            session in a ``finally``. Set to False for resume scenarios
            where the caller wants to preserve the container.

    Returns the SDK ``RunResult`` from ``Runner.run``. Raises if the
    sandbox bring-up fails or the run itself raises.
    """
    if scan_id is None:
        scan_id = f"scan-{uuid.uuid4().hex[:8]}"

    # Resolve run_dir before any heavy bring-up so the log file captures
    # everything from sandbox start onwards. Tracer (if present) owns the
    # canonical path; otherwise fall back to ``./strix_runs/<scan_id>``.
    run_dir = (
        tracer.get_run_dir()
        if tracer is not None and hasattr(tracer, "get_run_dir")
        else Path.cwd() / "strix_runs" / scan_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    teardown_logging = setup_scan_logging(run_dir)
    set_scan_id(scan_id)

    agents_path = run_dir / "agents.json"
    agents_db = run_dir / "agents.db"
    is_resume = agents_path.exists()

    lock_handle = _acquire_run_lock(run_dir)

    logger.info(
        "%s Strix scan %s (image=%s, max_turns=%d, interactive=%s, run_dir=%s)",
        "Resuming" if is_resume else "Starting",
        scan_id,
        image,
        max_turns,
        interactive,
        run_dir,
    )

    resolved_model = model or load_settings().llm.model
    if not resolved_model:
        _release_run_lock(lock_handle)
        raise RuntimeError(
            "No LLM model configured. Set STRIX_LLM env or pass model= to run_strix_scan().",
        )
    logger.info("LLM model resolved: %s", resolved_model)

    # Caller may pre-create the coordinator so it can route stop/chat
    # commands while the scan loop runs in another thread.
    if coordinator is None:
        coordinator = AgentCoordinator()
    coordinator.set_snapshot_path(agents_path)

    if tracer is not None and hasattr(tracer, "hydrate_from_run_dir"):
        tracer.hydrate_from_run_dir()

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
            _release_run_lock(lock_handle)
            raise RuntimeError(
                f"Cannot resume scan {scan_id}: agents.json is unreadable: {exc}",
            ) from exc
        if not agents_db.exists():
            _release_run_lock(lock_handle)
            raise RuntimeError(
                f"Cannot resume scan {scan_id}: missing SDK session database at {agents_db}",
            )
        await coordinator.restore(snap)
        for aid, parent in coordinator.parent_of.items():
            if parent is None:
                root_id = aid
                break
        if root_id is None:
            _release_run_lock(lock_handle)
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
        # Lazy: ``strix.interface`` pulls cli→tui→scan which would cycle.
        from strix.interface.utils import is_whitebox_scan

        scan_mode = str(scan_config.get("scan_mode") or "deep")
        is_whitebox = is_whitebox_scan(scan_config.get("targets") or [])
        skills = list(scan_config.get("skills") or [])
        diff_scope = scan_config.get("diff_scope") or None
        run_id = scan_config.get("run_id") or scan_id

        scope_context = _build_scope_context(scan_config)

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
                task=_build_root_task(scan_config),
                skills=skills,
                is_whitebox=is_whitebox,
                scan_mode=scan_mode,
                diff_scope=diff_scope,
            )

        agent_factory = make_child_factory(
            scan_mode=scan_mode,
            is_whitebox=is_whitebox,
            interactive=interactive,
            system_prompt_context=scope_context,
        )

        context: dict[str, Any] = {
            "coordinator": coordinator,
            "sandbox_session": bundle["session"],
            "sandbox_client": bundle["client"],
            "caido_client": bundle["caido_client"],
            "agent_id": root_id,
            "parent_id": None,
            "tracer": tracer,
            "model": resolved_model,
            "model_settings": None,
            "max_turns": max_turns,
            "agent_finish_called": False,
            "is_whitebox": is_whitebox,
            "interactive": interactive,
            "scan_mode": scan_mode,
            "diff_scope": diff_scope,
            "run_id": run_id,
            "agent_factory": agent_factory,
            "agents_db_path": agents_db,
            "_sessions_to_close": sessions_to_close,
        }

        reasoning_effort: Literal["low", "medium", "high"] | None = (
            load_settings().llm.reasoning_effort
        )
        model_settings = ModelSettings(
            parallel_tool_calls=False,
            tool_choice="required",
            retry=DEFAULT_RETRY,
        )
        if reasoning_effort is not None:
            model_settings = model_settings.resolve(
                ModelSettings(reasoning=Reasoning(effort=reasoning_effort)),
            )
        run_config = RunConfig(
            model=resolved_model,
            model_provider=build_multi_provider(),
            model_settings=model_settings,
            sandbox=SandboxRunConfig(client=bundle["client"], session=bundle["session"]),
            tracing_disabled=False,
            trace_include_sensitive_data=False,
        )

        if is_resume:
            await _respawn_subagents(
                coordinator=coordinator,
                agents_db_path=agents_db,
                factory=agent_factory,
                parent_ctx=context,
                resolved_model=resolved_model,
                reasoning_effort=reasoning_effort,
                root_id=root_id,
                sessions_to_close=sessions_to_close,
            )

        # All agents share one SQLite database; SDK session_id separates
        # each agent's conversation inside that database.
        root_session = open_agent_session(root_id, agents_db)
        sessions_to_close.append(root_session)
        await coordinator.attach_runtime(root_id, session=root_session)

        initial_input: Any = [] if is_resume else _build_root_task(scan_config)

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

        return await run_with_continuation(
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
        )
    except BaseException as exc:
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
            # here so the coordinator + tracer reflect reality, and stash the
            # error message for the status-line display.
            error_message = f"{type(exc).__name__}: {exc}"
            if tracer is not None and root_id in getattr(tracer, "agents", {}):
                tracer.agents[root_id]["status"] = "failed"
                tracer.agents[root_id]["error_message"] = error_message
                tracer.agents[root_id]["updated_at"] = datetime.now(UTC).isoformat()
            with contextlib.suppress(Exception):
                await coordinator.set_status(root_id, "failed")
            set_agent_id(None)
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
        _release_run_lock(lock_handle)
        logger.info("Strix scan %s done", scan_id)
        teardown_logging()


async def _respawn_subagents(
    *,
    coordinator: AgentCoordinator,
    agents_db_path: Path,
    factory: Any,
    parent_ctx: dict[str, Any],
    resolved_model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    root_id: str,
    sessions_to_close: list[SQLiteSession],
) -> None:
    """Re-spawn subagent runners from a restored coordinator snapshot.

    Each child gets its own SDK ``session_id`` inside the shared
    ``agents.db`` so the SDK replays its prior conversation. Interactive
    mode respawns every registered child as a parked runner unless it was
    actively running before the crash. That keeps completed/stopped/
    crashed/failed agents addressable: a later message wakes the SDK
    session instead of being dropped into a dead inbox.
    """
    interactive = bool(parent_ctx.get("interactive", False))
    async with coordinator._lock:
        # Snapshot the iteration view first so we can mutate via coordinator
        # below without "dict changed during iteration" trouble.
        agents_snapshot = [
            (aid, status, dict(coordinator.metadata.get(aid, {})))
            for aid, status in coordinator.statuses.items()
        ]
        candidates: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for aid, status, md in agents_snapshot:
            if not interactive and status not in {"running", "waiting", "llm_failed"}:
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

            child_session = open_agent_session(child_id, agents_db_path)
            sessions_to_close.append(child_session)
            await coordinator.attach_runtime(child_id, session=child_session)

            child_skills = list(md.get("skills") or [])
            child_agent = factory(name=name, skills=child_skills)

            child_ctx: dict[str, Any] = dict(parent_ctx)
            child_ctx["agent_id"] = child_id
            child_ctx["parent_id"] = parent_id
            child_ctx["agent_finish_called"] = False
            child_ctx["task"] = md.get("task", "")

            child_model_settings = ModelSettings(
                parallel_tool_calls=False,
                tool_choice="required",
                retry=DEFAULT_RETRY,
            )
            if reasoning_effort is not None:
                child_model_settings = child_model_settings.resolve(
                    ModelSettings(reasoning=Reasoning(effort=reasoning_effort)),
                )
            child_run_config = RunConfig(
                model=resolved_model,
                model_provider=build_multi_provider(),
                model_settings=child_model_settings,
                sandbox=SandboxRunConfig(
                    client=parent_ctx["sandbox_client"],
                    session=parent_ctx["sandbox_session"],
                ),
                tracing_disabled=False,
                trace_include_sensitive_data=False,
            )

            task_handle = asyncio.create_task(
                run_with_continuation(
                    agent=child_agent,
                    initial_input=[],
                    run_config=child_run_config,
                    context=child_ctx,
                    max_turns=int(parent_ctx.get("max_turns", 300)),
                    coordinator=coordinator,
                    agent_id=child_id,
                    interactive=bool(parent_ctx.get("interactive", False)),
                    session=child_session,
                    start_parked=start_parked,
                ),
                name=f"agent-{name}-{child_id}",
            )
            await coordinator.attach_runtime(child_id, task=task_handle)
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


def _acquire_run_lock(run_dir: Path) -> Any:
    """Take an exclusive flock on ``{run_dir}/.lock`` so two strix processes
    can't run on the same scan_id concurrently. Raises ``RuntimeError`` if
    another holder is detected. Best-effort on platforms without ``fcntl``.
    """
    lock_path = run_dir / ".lock"
    try:
        import fcntl
    except ImportError:
        return None
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"Another strix process appears to be running on this scan "
            f"(could not acquire lock at {lock_path}). Aborting.",
        ) from exc
    return handle


def _release_run_lock(handle: Any) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            handle.close()
