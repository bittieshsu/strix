"""Minimal in-container tool registry.

Used inside the sandbox container by ``strix.runtime.tool_server`` to
look up ``@register_tool``-decorated functions by name. Sandbox-bound
tools (browser, terminal, python, file_edit, proxy) live as
``*_actions.py`` modules with this decoration; the host POSTs to
:func:`tool_server.execute_tool` which dispatches via
:func:`get_tool_by_name`.

Host-side SDK function tools are wired through
:mod:`strix.agents.factory` and don't touch this registry.
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any


logger = logging.getLogger(__name__)


tools: list[dict[str, Any]] = []
_tools_by_name: dict[str, Callable[..., Any]] = {}


def _is_sandbox_mode() -> bool:
    return os.getenv("STRIX_SANDBOX_MODE", "false").lower() == "true"


def _is_browser_disabled() -> bool:
    return os.getenv("STRIX_DISABLE_BROWSER", "").lower() == "true"


def _has_perplexity_api() -> bool:
    return bool(os.getenv("PERPLEXITY_API_KEY"))


def _should_register_tool(
    *,
    sandbox_execution: bool,
    requires_browser_mode: bool,
    requires_web_search_mode: bool,
) -> bool:
    """In-container side only registers sandbox-execution tools."""
    sandbox_mode = _is_sandbox_mode()

    if sandbox_mode and not sandbox_execution:
        return False
    if requires_browser_mode and _is_browser_disabled():
        return False
    return not (requires_web_search_mode and not _has_perplexity_api())


def register_tool(
    func: Callable[..., Any] | None = None,
    *,
    sandbox_execution: bool = True,
    requires_browser_mode: bool = False,
    requires_web_search_mode: bool = False,
) -> Callable[..., Any]:
    """Register a tool function for in-container dispatch.

    Decorations are conditional on the env (``STRIX_SANDBOX_MODE``,
    ``STRIX_DISABLE_BROWSER``, ``PERPLEXITY_API_KEY``) so the host
    side, which imports these modules but doesn't run sandbox-bound
    tools locally, doesn't accumulate dead registrations.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        if not _should_register_tool(
            sandbox_execution=sandbox_execution,
            requires_browser_mode=requires_browser_mode,
            requires_web_search_mode=requires_web_search_mode,
        ):
            return f

        tools.append(
            {
                "name": f.__name__,
                "function": f,
                "sandbox_execution": sandbox_execution,
            },
        )
        _tools_by_name[f.__name__] = f

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def get_tool_by_name(name: str) -> Callable[..., Any] | None:
    return _tools_by_name.get(name)


def get_tool_names() -> list[str]:
    return list(_tools_by_name.keys())


def clear_registry() -> None:
    tools.clear()
    _tools_by_name.clear()
