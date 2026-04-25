"""Phase 2.1 smoke tests for the sandbox dispatch helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from strix.tools._sandbox_dispatch import post_to_sandbox


@dataclass
class _Ctx:
    """Stand-in for ``RunContextWrapper``. Only ``.context`` is touched."""

    context: dict[str, Any] = field(default_factory=dict)


def _ok_ctx(**overrides: Any) -> _Ctx:
    base: dict[str, Any] = {
        "tool_server_host_port": 48081,
        "sandbox_token": "test-bearer",
        "agent_id": "agent-1",
    }
    base.update(overrides)
    return _Ctx(context=base)


@pytest.mark.asyncio
async def test_missing_context_returns_error() -> None:
    """If ``ctx.context`` isn't a dict, return error — never raise."""
    ctx = _Ctx(context="not a dict")
    result = await post_to_sandbox(ctx, "browser_action", {"action": "launch"})
    assert "error" in result
    assert "context" in result["error"].lower()


@pytest.mark.asyncio
async def test_missing_port_or_token_returns_error() -> None:
    ctx = _Ctx(context={"sandbox_token": "tok"})  # no port
    result = await post_to_sandbox(ctx, "x", {})
    assert "error" in result
    assert "tool server port" in result["error"].lower()


@pytest.mark.asyncio
async def test_successful_response_returned_as_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(
            status_code=200,
            json={"result": "ok"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await post_to_sandbox(_ok_ctx(), "terminal_execute", {"command": "ls"})
    assert result == {"result": "ok"}
    assert captured["url"] == "http://127.0.0.1:48081/execute"
    assert captured["json"] == {
        "agent_id": "agent-1",
        "tool_name": "terminal_execute",
        "kwargs": {"command": "ls"},
    }
    assert captured["headers"]["Authorization"] == "Bearer test-bearer"


@pytest.mark.asyncio
async def test_401_returns_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        **_: Any,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            text="forbidden",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "x", {})
    assert "error" in result
    assert "authorization" in result["error"].lower()


@pytest.mark.asyncio
async def test_5xx_returns_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        **_: Any,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            text="server fell over",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "browser_action", {})
    assert "error" in result
    assert "503" in result["error"]


@pytest.mark.asyncio
async def test_timeout_returns_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "python_action", {})
    assert "error" in result
    assert "timed out" in result["error"].lower()


@pytest.mark.asyncio
async def test_connection_error_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "x", {})
    assert "error" in result
    assert "connection" in result["error"].lower()


@pytest.mark.asyncio
async def test_response_too_large_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """C18 (AUDIT_R3): response > 50MB returns error, doesn't OOM."""

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        **_: Any,
    ) -> httpx.Response:
        # Construct a response with a >50MB body. Use a string trick —
        # httpx.Response stores .content directly; we make it look huge.
        big = b"x" * (51 * 1024 * 1024)
        return httpx.Response(
            status_code=200,
            content=big,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "browser_action", {})
    assert "error" in result
    assert "too large" in result["error"].lower()


@pytest.mark.asyncio
async def test_non_json_response_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        **_: Any,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="hello not json",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "x", {})
    assert "error" in result
    assert "non-json" in result["error"].lower()


@pytest.mark.asyncio
async def test_non_object_json_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        **_: Any,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=["a", "list", "not", "an", "object"],
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await post_to_sandbox(_ok_ctx(), "x", {})
    assert "error" in result
    assert "non-object" in result["error"].lower()
