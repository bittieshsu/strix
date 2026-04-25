"""Phase 5 tests for the SDK agent factory + prompt renderer.

These two modules are the keystone wiring between Phases 2-4 and an
actual ``Runner.run`` invocation. The tests verify:

- The prompt renderer reuses the existing Jinja template (parity with
  legacy LLM._load_system_prompt) and degrades gracefully when the
  template isn't available.
- ``build_strix_agent(is_root=True)`` carries ``finish_scan`` and
  stops on it; child agents carry ``agent_finish`` and stop on it.
- ``make_child_factory`` snapshots scan-level config into a closure
  so each spawned child inherits the right scan_mode / is_whitebox /
  prompt context without create_agent having to re-derive it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agents import Agent
from agents.tool import FunctionTool

from strix.agents.factory import build_strix_agent, make_child_factory
from strix.agents.prompt import _resolve_skills, render_system_prompt


# --- prompt renderer ----------------------------------------------------


def test_resolve_skills_deduplicates_and_orders() -> None:
    out = _resolve_skills(
        requested=["recon", "xss", "recon"],
        scan_mode="deep",
        is_whitebox=False,
    )
    assert out == ["recon", "xss", "scan_modes/deep"]


def test_resolve_skills_adds_whitebox_pair() -> None:
    out = _resolve_skills(requested=None, scan_mode="fast", is_whitebox=True)
    # The whitebox pair sits at the tail; scan_modes goes in the middle
    # because callers can append more skills after it via the requested arg.
    assert out == [
        "scan_modes/fast",
        "coordination/source_aware_whitebox",
        "custom/source_aware_sast",
    ]


def test_render_system_prompt_returns_string() -> None:
    """Smoke: the StrixAgent template is on disk and renders to non-empty."""
    out = render_system_prompt(skills=[], scan_mode="deep")
    assert isinstance(out, str)
    # The first line of the template starts with 'You are Strix'.
    assert out.startswith("You are Strix")


def test_render_system_prompt_swallows_template_errors() -> None:
    """If the template path can't be resolved, return an empty string
    (not raise) — agent construction must never blow up on prompt load."""
    with patch(
        "strix.agents.prompt.get_strix_resource_path",
        side_effect=RuntimeError("missing"),
    ):
        out = render_system_prompt(skills=[])
    assert out == ""


# --- factory: shape + tools --------------------------------------------


def test_root_agent_carries_finish_scan_and_stops_there() -> None:
    agent = build_strix_agent(name="strix", is_root=True)
    assert isinstance(agent, Agent)
    tool_names = {t.name for t in agent.tools if isinstance(t, FunctionTool)}
    assert "finish_scan" in tool_names
    assert "agent_finish" not in tool_names
    behavior = agent.tool_use_behavior
    # StopAtTools is a TypedDict at runtime → behavior is a dict.
    assert isinstance(behavior, dict)
    assert behavior["stop_at_tool_names"] == ["finish_scan"]


def test_child_agent_carries_agent_finish_and_stops_there() -> None:
    agent = build_strix_agent(name="recon-bot", is_root=False)
    tool_names = {t.name for t in agent.tools if isinstance(t, FunctionTool)}
    assert "agent_finish" in tool_names
    assert "finish_scan" not in tool_names
    behavior = agent.tool_use_behavior
    assert isinstance(behavior, dict)
    assert behavior["stop_at_tool_names"] == ["agent_finish"]


def test_root_and_child_share_base_tool_set() -> None:
    """The base tool set (think/todo/notes/file_edit/web_search/etc) is
    identical between root and child — only the terminator differs."""
    root = build_strix_agent(is_root=True)
    child = build_strix_agent(is_root=False)
    root_names = {t.name for t in root.tools if isinstance(t, FunctionTool)}
    child_names = {t.name for t in child.tools if isinstance(t, FunctionTool)}
    # Drop the terminators and compare.
    assert root_names - {"finish_scan"} == child_names - {"agent_finish"}


def test_agent_includes_graph_and_sandbox_tools() -> None:
    """The graph + sandbox tool families are required for parity with
    legacy. Spot-check the ones most likely to be forgotten in a refactor."""
    agent = build_strix_agent(is_root=True)
    names = {t.name for t in agent.tools if isinstance(t, FunctionTool)}
    expected = {
        "think",
        "create_todo",
        "create_note",
        "web_search",
        "str_replace_editor",
        "create_vulnerability_report",
        "load_skill",
        "browser_action",
        "terminal_execute",
        "python_action",
        "view_agent_graph",
        "agent_status",
        "send_message_to_agent",
        "wait_for_message",
        "create_agent",
    }
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


def test_agent_does_not_include_caido_tools() -> None:
    """Caido tools come from CaidoCapability.tools(); the agent doesn't
    declare them directly to avoid double-registration when the SDK
    runtime merges capability tools."""
    agent = build_strix_agent(is_root=True)
    names = {t.name for t in agent.tools if isinstance(t, FunctionTool)}
    caido = {
        "list_requests",
        "view_request",
        "send_request",
        "repeat_request",
        "scope_rules",
        "list_sitemap",
        "view_sitemap_entry",
    }
    overlap = names & caido
    assert overlap == set(), f"unexpected Caido tools in agent.tools: {overlap}"


def test_agent_uses_run_config_model() -> None:
    """``model=None`` so the RunConfig drives the model alias through
    MultiProvider rather than an SDK default like gpt-4.1."""
    agent = build_strix_agent(is_root=True)
    assert agent.model is None


def test_agent_instructions_contain_rendered_prompt() -> None:
    """The factory must wire the rendered prompt into ``instructions``."""
    agent = build_strix_agent(is_root=True, scan_mode="deep")
    assert isinstance(agent.instructions, str)
    assert agent.instructions.startswith("You are Strix")


# --- child factory ------------------------------------------------------


def test_make_child_factory_returns_callable_that_builds_child() -> None:
    factory = make_child_factory(scan_mode="deep", is_whitebox=False)
    assert callable(factory)
    child = factory(name="sub-1", skills=["recon"])
    assert isinstance(child, Agent)
    assert child.name == "sub-1"
    behavior = child.tool_use_behavior
    assert isinstance(behavior, dict)
    assert behavior["stop_at_tool_names"] == ["agent_finish"]


def test_make_child_factory_passes_scan_level_config() -> None:
    """Verify scan_mode + is_whitebox flow into the rendered prompt
    via the closure rather than the create_agent call site."""
    captured: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-prompt"

    factory = make_child_factory(
        scan_mode="fast",
        is_whitebox=True,
        interactive=True,
        system_prompt_context={"scope_source": "test"},
    )
    with patch("strix.agents.factory.render_system_prompt", side_effect=fake_render):
        factory(name="child", skills=["xss"])

    assert captured["scan_mode"] == "fast"
    assert captured["is_whitebox"] is True
    assert captured["interactive"] is True
    assert captured["system_prompt_context"] == {"scope_source": "test"}
    assert captured["skills"] == ["xss"]
