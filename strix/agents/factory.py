"""build_strix_agent — assemble an ``agents.Agent`` for root or child runs.

This is the keystone that links Phase 2's SDK function tools, Phase 3's
graph tools, Phase 4's CaidoCapability, and the rendered Jinja prompt
from :mod:`strix.agents.prompt` into a single ``agents.Agent``
instance ready for ``Runner.run``.

Two flavors:

- **Root** (``is_root=True``): the top-level scan agent. Carries
  ``finish_scan`` (terminates the scan), no ``agent_finish`` (that's
  for subagents). ``tool_use_behavior`` stops on ``finish_scan`` so
  the model can't accidentally keep talking after marking the scan
  complete.

- **Child** (``is_root=False``): subagents spawned by the
  ``create_agent`` graph tool. Carries ``agent_finish``, no
  ``finish_scan``. ``tool_use_behavior`` stops on ``agent_finish``
  (C4 — without this, the SDK loop would keep going to ``max_turns``
  even after the child reported back to its parent).

Caido tools come from ``CaidoCapability.tools()`` automatically via
the SDK's capability merge — we don't include them here. Skills are
injected via the prompt; the model can also load more at runtime via
the ``load_skill`` tool.

References:
    - PLAYBOOK.md §4.3 (graph tool wiring)
    - AUDIT.md §2.4 (C4 — stop_at_tool_names is required for subagents)
"""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent
from agents.agent import StopAtTools
from agents.tool import Tool

from strix.agents.prompt import render_system_prompt
from strix.tools.agents_graph.tools import (
    agent_finish,
    agent_status,
    create_agent,
    send_message_to_agent,
    view_agent_graph,
    wait_for_message,
)
from strix.tools.browser.tool import browser_action
from strix.tools.file_edit.tools import (
    list_files,
    search_files,
    str_replace_editor,
)
from strix.tools.finish.tool import finish_scan
from strix.tools.load_skill.tool import load_skill
from strix.tools.notes.tools import (
    create_note,
    delete_note,
    get_note,
    list_notes,
    update_note,
)
from strix.tools.python.tool import python_action
from strix.tools.reporting.tool import create_vulnerability_report
from strix.tools.terminal.tool import terminal_execute
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


logger = logging.getLogger(__name__)


# Tools every Strix agent has, root or child. The Caido proxy tools
# (list_requests, view_request, send_request, ...) are NOT here —
# CaidoCapability.tools() returns them and the SDK merges them in.
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
    # File edit (sandbox-bound)
    str_replace_editor,
    list_files,
    search_files,
    # Reporting
    create_vulnerability_report,
    # Skill loading
    load_skill,
    # Sandbox primitives
    browser_action,
    terminal_execute,
    python_action,
    # Multi-agent graph tools (the bus is in ctx.context)
    view_agent_graph,
    agent_status,
    send_message_to_agent,
    wait_for_message,
    create_agent,
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
) -> Agent[Any]:
    """Build an ``agents.Agent`` configured for either root or child use.

    Args:
        name: Agent name. Surfaces in traces and the bus's ``names`` map.
            Defaults to ``"strix"`` for the root; create_agent passes
            distinct names per child.
        skills: Skills to preload into the system prompt. The agent can
            also load more at runtime via the ``load_skill`` tool.
        is_root: Selects the tool list and ``tool_use_behavior``.
            Root carries ``finish_scan`` and stops there; child carries
            ``agent_finish`` and stops there.
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

    Returns the ``Agent`` instance with ``model=None`` so the
    ``RunConfig.model`` (built by ``make_run_config``) drives provider
    selection. ``agents.Agent`` is generic on context type; we let
    the caller's ``Runner.run(context=...)`` typing determine that.
    """
    instructions = render_system_prompt(
        skills=skills,
        scan_mode=scan_mode,
        is_whitebox=is_whitebox,
        interactive=interactive,
        system_prompt_context=system_prompt_context,
    )

    # Tool list + termination tool depend on is_root. The tuple-then-
    # list dance keeps _BASE_TOOLS immutable so concurrent agent builds
    # can't accidentally mutate each other's tool list.
    if is_root:
        tools: list[Tool] = [*_BASE_TOOLS, finish_scan]
        stop_at = ("finish_scan",)
    else:
        tools = [*_BASE_TOOLS, agent_finish]
        stop_at = ("agent_finish",)

    return Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        tool_use_behavior=StopAtTools(stop_at_tool_names=list(stop_at)),
        # model=None so ``RunConfig.model`` (e.g. ``strix/claude-sonnet-4.6``)
        # routes through MultiProvider rather than the SDK's default.
        model=None,
    )


def make_child_factory(
    *,
    scan_mode: str = "deep",
    is_whitebox: bool = False,
    interactive: bool = False,
    system_prompt_context: dict[str, Any] | None = None,
) -> Any:
    """Return a callable suitable for ``ctx.context['agent_factory']``.

    The Phase 3 ``create_agent`` graph tool reads
    ``ctx.context['agent_factory']`` and calls it with ``name=`` and
    ``skills=`` to build a child Agent. We snapshot the run-level
    arguments (scan_mode, is_whitebox, etc.) into a closure so each
    child inherits the right scan-level configuration without the
    create_agent tool having to know about them.
    """

    def _factory(*, name: str, skills: list[str]) -> Agent[Any]:
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
