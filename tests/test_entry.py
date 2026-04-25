"""Phase 5 tests for the top-level SDK scan entry point.

We never spin up a real Docker container or hit a real LLM here. The
tests patch ``session_manager.create_or_reuse``, ``Runner.run``, and
the agent factory so we can verify the wiring shape:

- The bus is registered with a root agent before Runner.run.
- The context dict carries every field downstream code (tools, hooks,
  filter) reads.
- The session manager's bundle flows through to the context (host
  ports, bearer, sandbox session/client).
- ``cleanup_on_exit=True`` always cleans up, even when Runner.run
  raises.
- ``cleanup_on_exit=False`` preserves the cached session.
- Cancellation propagates: if Runner.run raises, descendants are
  cancelled before re-raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strix.entry import _build_root_task, _build_scope_context, run_strix_scan
from strix.orchestration.bus import AgentMessageBus


# --- helpers ------------------------------------------------------------


def _bundle_for_test() -> dict[str, Any]:
    return {
        "client": MagicMock(name="docker_client"),
        "session": MagicMock(name="sandbox_session"),
        "capability": MagicMock(),
        "tool_server_host_port": 12001,
        "caido_host_port": 12002,
        "bearer": "test-bearer-token-1234567890",
    }


def _scan_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "targets": [
            {
                "type": "web_application",
                "details": {"target_url": "https://example.com"},
            },
        ],
        "user_instructions": "find xss",
        "scan_mode": "deep",
        "is_whitebox": False,
    }
    base.update(overrides)
    return base


# --- task / scope builders ---------------------------------------------


def test_build_root_task_groups_targets_and_appends_instructions() -> None:
    config = _scan_config(
        targets=[
            {
                "type": "repository",
                "details": {
                    "target_repo": "https://github.com/x/y",
                    "cloned_repo_path": "/tmp/y",
                    "workspace_subdir": "y",
                },
            },
            {
                "type": "ip_address",
                "details": {"target_ip": "10.0.0.1"},
            },
        ],
        user_instructions="report only critical issues",
    )
    task = _build_root_task(config)
    assert "Repositories:" in task
    assert "https://github.com/x/y (available at: /workspace/y)" in task
    assert "IP Addresses:" in task
    assert "10.0.0.1" in task
    assert "Special instructions: report only critical issues" in task


def test_build_root_task_renders_diff_scope_block() -> None:
    config = _scan_config(
        diff_scope={
            "active": True,
            "repos": [
                {
                    "workspace_subdir": "service-x",
                    "analyzable_files_count": 7,
                    "deleted_files_count": 2,
                },
            ],
        },
    )
    task = _build_root_task(config)
    assert "Scope Constraints:" in task
    assert "service-x: 7 changed file(s)" in task
    assert "service-x: 2 deleted file(s)" in task


def test_build_scope_context_marks_authorization_source() -> None:
    config = _scan_config(
        targets=[
            {
                "type": "web_application",
                "details": {"target_url": "https://target.test"},
            },
        ],
    )
    ctx = _build_scope_context(config)
    assert ctx["scope_source"] == "system_scan_config"
    assert ctx["authorization_source"] == "strix_platform_verified_targets"
    assert ctx["user_instructions_do_not_expand_scope"] is True
    assert ctx["authorized_targets"] == [
        {"type": "web_application", "value": "https://target.test", "workspace_path": ""},
    ]


# --- run_strix_scan wiring ---------------------------------------------


@pytest.mark.asyncio
async def test_run_strix_scan_wires_context_and_calls_runner(tmp_path: Path) -> None:
    """End-to-end (mocked) — assert every downstream consumer of context
    sees the bundle's bearer + host ports."""
    bundle = _bundle_for_test()
    captured_context: dict[str, Any] = {}

    async def fake_runner_run(*args: Any, **kwargs: Any) -> Any:
        captured_context.update(kwargs.get("context", {}))
        return MagicMock(name="run_result")

    with (
        patch(
            "strix.entry.session_manager.create_or_reuse",
            new=AsyncMock(return_value=bundle),
        ) as create_mock,
        patch(
            "strix.entry.session_manager.cleanup",
            new=AsyncMock(),
        ) as cleanup_mock,
        patch("strix.entry.Runner.run", side_effect=fake_runner_run) as runner_mock,
        # Stub the factory to avoid rendering the 158k-char prompt for
        # every test (it's covered by sdk_prompt tests).
        patch(
            "strix.entry.build_strix_agent",
            return_value=MagicMock(name="root_agent"),
        ) as factory_mock,
    ):
        await run_strix_scan(
            scan_config=_scan_config(),
            scan_id="scan-test",
            image="strix-sandbox:test",
            sources_path=tmp_path,
        )

    # Session manager calls.
    create_mock.assert_awaited_once()
    create_args = create_mock.await_args
    assert create_args is not None
    assert create_args.args == ("scan-test",)
    assert create_args.kwargs["image"] == "strix-sandbox:test"
    assert create_args.kwargs["sources_path"] == tmp_path
    cleanup_mock.assert_awaited_once_with("scan-test")

    # Factory called with is_root=True.
    factory_mock.assert_called_once()
    assert factory_mock.call_args.kwargs["is_root"] is True

    # Runner.run called once with the root agent.
    assert runner_mock.call_count == 1

    # Context shape passed into Runner.run.
    assert captured_context["sandbox_session"] is bundle["session"]
    assert captured_context["sandbox_client"] is bundle["client"]
    assert captured_context["sandbox_token"] == bundle["bearer"]
    assert captured_context["tool_server_host_port"] == bundle["tool_server_host_port"]
    assert captured_context["caido_host_port"] == bundle["caido_host_port"]
    # Bus is registered and root agent_id is populated.
    bus = captured_context["bus"]
    assert isinstance(bus, AgentMessageBus)
    assert captured_context["agent_id"] in bus.statuses
    assert bus.parent_of[captured_context["agent_id"]] is None
    # Child factory wired through so create_agent works.
    assert callable(captured_context["agent_factory"])


@pytest.mark.asyncio
async def test_run_strix_scan_cleans_up_on_runner_failure(tmp_path: Path) -> None:
    """If Runner.run raises, cleanup must still fire (the finally branch)."""
    bundle = _bundle_for_test()

    with (
        patch(
            "strix.entry.session_manager.create_or_reuse",
            new=AsyncMock(return_value=bundle),
        ),
        patch(
            "strix.entry.session_manager.cleanup",
            new=AsyncMock(),
        ) as cleanup_mock,
        patch(
            "strix.entry.Runner.run",
            side_effect=RuntimeError("simulated LLM blow-up"),
        ),
        patch("strix.entry.build_strix_agent", return_value=MagicMock()),
        pytest.raises(RuntimeError, match="simulated LLM"),
    ):
        await run_strix_scan(
            scan_config=_scan_config(),
            scan_id="scan-fail",
            image="i",
            sources_path=tmp_path,
        )

    cleanup_mock.assert_awaited_once_with("scan-fail")


@pytest.mark.asyncio
async def test_run_strix_scan_skips_cleanup_when_disabled(tmp_path: Path) -> None:
    bundle = _bundle_for_test()

    async def fake_runner_run(*args: Any, **kwargs: Any) -> Any:
        return MagicMock()

    with (
        patch(
            "strix.entry.session_manager.create_or_reuse",
            new=AsyncMock(return_value=bundle),
        ),
        patch(
            "strix.entry.session_manager.cleanup",
            new=AsyncMock(),
        ) as cleanup_mock,
        patch("strix.entry.Runner.run", side_effect=fake_runner_run),
        patch("strix.entry.build_strix_agent", return_value=MagicMock()),
    ):
        await run_strix_scan(
            scan_config=_scan_config(),
            scan_id="scan-keep",
            image="i",
            sources_path=tmp_path,
            cleanup_on_exit=False,
        )

    cleanup_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_strix_scan_auto_generates_scan_id(tmp_path: Path) -> None:
    """A caller without a stable id should still get a valid scan_id
    flowing into create_or_reuse."""
    bundle = _bundle_for_test()
    captured_scan_id: list[str] = []

    async def fake_create(scan_id: str, **_kwargs: Any) -> Any:
        captured_scan_id.append(scan_id)
        return bundle

    with (
        patch(
            "strix.entry.session_manager.create_or_reuse",
            new=AsyncMock(side_effect=fake_create),
        ),
        patch("strix.entry.session_manager.cleanup", new=AsyncMock()),
        patch("strix.entry.Runner.run", new=AsyncMock(return_value=MagicMock())),
        patch("strix.entry.build_strix_agent", return_value=MagicMock()),
    ):
        await run_strix_scan(
            scan_config=_scan_config(),
            image="i",
            sources_path=tmp_path,
        )

    assert len(captured_scan_id) == 1
    assert captured_scan_id[0].startswith("scan-")
    assert len(captured_scan_id[0]) > len("scan-")


@pytest.mark.asyncio
async def test_run_strix_scan_passes_scan_level_config_into_factory(
    tmp_path: Path,
) -> None:
    """scan_mode / is_whitebox flow from scan_config into both the
    root factory call and the child factory closure."""
    bundle = _bundle_for_test()
    factory_calls: list[dict[str, Any]] = []

    def fake_factory(**kwargs: Any) -> Any:
        factory_calls.append(kwargs)
        return MagicMock()

    with (
        patch(
            "strix.entry.session_manager.create_or_reuse",
            new=AsyncMock(return_value=bundle),
        ),
        patch("strix.entry.session_manager.cleanup", new=AsyncMock()),
        patch("strix.entry.Runner.run", new=AsyncMock(return_value=MagicMock())),
        patch("strix.entry.build_strix_agent", side_effect=fake_factory),
    ):
        await run_strix_scan(
            scan_config=_scan_config(scan_mode="fast", is_whitebox=True),
            scan_id="s",
            image="i",
            sources_path=tmp_path,
        )

    assert factory_calls[0]["scan_mode"] == "fast"
    assert factory_calls[0]["is_whitebox"] is True
    assert factory_calls[0]["is_root"] is True
