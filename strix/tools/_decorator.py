"""``strix_tool`` — ``function_tool`` factory with Strix defaults.

Every tool uses ``@strix_tool`` instead of bare ``@function_tool`` so
defaults stay consistent across the suite. Override per call when
needed.

Defaults:
    - ``timeout``: 120s.
    - ``timeout_behavior``: ``"error_as_result"`` for idempotent tools.
      Critical sandbox tools (terminal, browser, python) opt into
      ``timeout_behavior="raise_exception"`` explicitly so the SDK
      fails the run rather than letting the model retry a hung call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from agents import function_tool
from agents.tool import FunctionTool


_ToolFn = Callable[..., Any]
_ToolBehavior = Literal["error_as_result", "raise_exception"]


def strix_tool(
    *,
    timeout: float = 120.0,
    timeout_behavior: _ToolBehavior = "error_as_result",
    name_override: str | None = None,
    description_override: str | None = None,
    strict_mode: bool = True,
) -> Callable[[_ToolFn], FunctionTool]:
    """Wrap ``agents.function_tool`` with Strix defaults.

    Strict mode is on by default (forbids free-form ``dict[str, X]``
    parameters because the strict JSON schema needs
    ``additionalProperties: false``). A few tools that take arbitrary
    header / modification dicts opt out via ``strict_mode=False``.

    Usage::

        @strix_tool()
        async def my_tool(ctx: RunContextWrapper, x: int) -> str: ...

        @strix_tool(timeout=300, timeout_behavior="raise_exception")
        async def critical_tool(ctx: RunContextWrapper, ...) -> str: ...

        @strix_tool(strict_mode=False)
        async def free_form_dict_tool(
            ctx: RunContextWrapper, headers: dict[str, str],
        ) -> str: ...
    """
    return function_tool(
        timeout=timeout,
        timeout_behavior=timeout_behavior,
        name_override=name_override,
        description_override=description_override,
        strict_mode=strict_mode,
    )
