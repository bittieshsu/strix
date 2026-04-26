"""``build_strix_agent`` — assemble an ``agents.Agent`` for root or child.

Wires the SDK function tools, multi-agent graph tools, and the rendered
Jinja prompt into one ``agents.Agent`` ready for ``Runner.run``.

Two flavors:

- **Root** (``is_root=True``): top-level scan agent. Carries
  ``finish_scan`` and stops after that tool reports ``scan_completed``.
- **Child** (``is_root=False``): subagents spawned by the
  ``create_agent`` graph tool. Carries ``agent_finish`` and stops
  after that tool reports ``agent_completed``.

Skills are baked into the system prompt at scan bring-up; there's no
runtime skill-loading tool.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents.agent import ToolsToFinalOutputResult
from agents.sandbox import SandboxAgent
from agents.sandbox.capabilities import Filesystem, Shell
from agents.tool import Tool

from strix.agents.prompt import render_system_prompt
from strix.tools.agents_graph.tools import (
    agent_finish,
    create_agent,
    send_message_to_agent,
    stop_agent,
    view_agent_graph,
    wait_for_message,
)
from strix.tools.finish.tool import finish_scan
from strix.tools.notes.tools import (
    create_note,
    delete_note,
    get_note,
    list_notes,
    update_note,
)
from strix.tools.proxy.tools import (
    list_requests,
    repeat_request,
    scope_rules,
    send_request,
    view_request,
)
from strix.tools.python.tool import python_action
from strix.tools.reporting.tool import create_vulnerability_report
from strix.tools.thinking.tool import think
from strix.tools.todo.tools import (
    create_todo,
    delete_todo,
    list_todos,
    mark_todo_done,
    mark_todo_pending,
    update_todo,
)
from strix.tools.web_search.tool import web_search


if TYPE_CHECKING:
    from agents import RunContextWrapper
    from agents.tool import FunctionToolResult


logger = logging.getLogger(__name__)


def _lifecycle_tool_completed(tool_name: str, output: Any) -> bool:
    if tool_name == "agent_finish":
        completion_key = "agent_completed"
    elif tool_name == "finish_scan":
        completion_key = "scan_completed"
    else:
        return False

    if not isinstance(output, str):
        return False
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return False
    return bool(isinstance(parsed, dict) and parsed.get("success") and parsed.get(completion_key))


def _wait_tool_parked(tool_name: str, output: Any) -> bool:
    if tool_name != "wait_for_message" or not isinstance(output, str):
        return False
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return False
    return bool(
        isinstance(parsed, dict)
        and parsed.get("success")
        and parsed.get("agent_waiting")
        and parsed.get("status") == "waiting"
    )


def _finish_tool_use_behavior(
    ctx: RunContextWrapper[Any],
    tool_results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """Stop only after a lifecycle tool reports successful completion."""
    interactive = (
        bool(ctx.context.get("interactive", False)) if isinstance(ctx.context, dict) else False
    )
    for tool_result in tool_results:
        if _lifecycle_tool_completed(tool_result.tool.name, tool_result.output):
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=tool_result.output,
            )
        if interactive and _wait_tool_parked(tool_result.tool.name, tool_result.output):
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=tool_result.output,
            )
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


# Host-side Strix tools. Sandbox shell + filesystem are added per-run
# by the SDK via the ``Shell`` and ``Filesystem`` capabilities below
# (they bind to the live sandbox session and emit ``exec_command`` /
# ``write_stdin`` / ``apply_patch`` / ``view_image`` function tools).
_BASE_TOOLS: tuple[Tool, ...] = (
    # Thinking + planning
    think,
    # Per-agent todos
    create_todo,
    list_todos,
    update_todo,
    mark_todo_done,
    mark_todo_pending,
    delete_todo,
    # Shared notes (per-run JSONL store)
    create_note,
    list_notes,
    get_note,
    update_note,
    delete_note,
    # Web search (only registered if PERPLEXITY_API_KEY is set; the
    # tool itself returns a structured error when not configured, so
    # always exposing it is safe)
    web_search,
    # Reporting
    create_vulnerability_report,
    # Caido HTTP/HTTPS proxy
    list_requests,
    view_request,
    send_request,
    repeat_request,
    scope_rules,
    # Stateless Python execution with proxy helpers pre-bound
    python_action,
    # Multi-agent graph tools (the coordinator is in ctx.context)
    view_agent_graph,
    send_message_to_agent,
    wait_for_message,
    create_agent,
    stop_agent,
)


def build_strix_agent(
    *,
    name: str = "strix",
    skills: list[str] | None = None,
    is_root: bool,
    scan_mode: str = "deep",
    is_whitebox: bool = False,
    interactive: bool = False,
    system_prompt_context: dict[str, Any] | None = None,
) -> SandboxAgent[Any]:
    """Build a ``SandboxAgent`` configured for either root or child use.

    The ``Shell`` and ``Filesystem`` capabilities are added unbound; the
    SDK's runtime binds them per-run against the live sandbox session
    set on ``RunConfig.sandbox`` and merges their tools (``exec_command``,
    ``write_stdin``, ``apply_patch``, ``view_image``) into the agent's
    final tool list. We deliberately exclude ``Compaction`` (OpenAI
    Responses API only).

    Args:
        name: Agent name. Surfaces in traces and the coordinator's ``names`` map.
            Defaults to ``"strix"`` for the root; create_agent passes
            distinct names per child.
        skills: Skills to preload into the system prompt.
        is_root: Selects the tool list and ``tool_use_behavior``.
            Root carries ``finish_scan`` and child carries ``agent_finish``;
            the run only stops when the lifecycle tool result succeeds.
        scan_mode: ``"deep"`` etc.; routes the scan-mode skill section
            of the prompt template.
        is_whitebox: Whitebox source-aware mode toggle. Adds two extra
            skills to the prompt and gates whitebox-only behavior in
            the create_agent / wiki integration.
        interactive: Renders the interactive-mode communication block
            in the system prompt.
        system_prompt_context: Free-form dict the prompt template
            renders into the ``system_prompt_context`` variable —
            today carries the scan scope / authorization block.
    """
    instructions = render_system_prompt(
        skills=skills,
        scan_mode=scan_mode,
        is_whitebox=is_whitebox,
        is_root=is_root,
        interactive=interactive,
        system_prompt_context=system_prompt_context,
    )

    if is_root:
        tools: list[Tool] = [*_BASE_TOOLS, finish_scan]
    else:
        tools = [*_BASE_TOOLS, agent_finish]

    logger.info(
        "Built %s agent '%s' (skills=%d, tools=%d, scan_mode=%s, whitebox=%s)",
        "root" if is_root else "child",
        name,
        len(skills or []),
        len(tools),
        scan_mode,
        is_whitebox,
    )

    return SandboxAgent(
        name=name,
        instructions=instructions,
        tools=tools,
        tool_use_behavior=_finish_tool_use_behavior,
        # model=None so ``RunConfig.model`` drives provider selection
        # via :func:`build_multi_provider` rather than the SDK's default.
        model=None,
        capabilities=[Filesystem(), Shell()],
    )


def make_child_factory(
    *,
    scan_mode: str = "deep",
    is_whitebox: bool = False,
    interactive: bool = False,
    system_prompt_context: dict[str, Any] | None = None,
) -> Any:
    """Return the runner-owned builder used by ``spawn_child_agent``.

    Run-level arguments (``scan_mode``, ``is_whitebox``, etc.) are
    captured in a closure so each child inherits scan-level configuration
    without the graph tool knowing about runner internals.
    """

    def _factory(*, name: str, skills: list[str]) -> SandboxAgent[Any]:
        return build_strix_agent(
            name=name,
            skills=skills,
            is_root=False,
            scan_mode=scan_mode,
            is_whitebox=is_whitebox,
            interactive=interactive,
            system_prompt_context=system_prompt_context,
        )

    return _factory
