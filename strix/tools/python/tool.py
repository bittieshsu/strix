"""SDK function-tool wrapper for the legacy ``python_action`` tool.

Sandbox-bound. The in-container manager keeps long-lived IPython
sessions keyed by ``session_id`` so the model can build up state
across multiple ``execute`` calls. Pure pass-through wrapper.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._sandbox_dispatch import post_to_sandbox


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


PythonAction = Literal["new_session", "execute", "close", "list_sessions"]


@strix_tool(timeout=180)
async def python_action(
    ctx: RunContextWrapper,
    action: PythonAction,
    code: str | None = None,
    timeout: int = 30,
    session_id: str | None = None,
) -> str:
    """Manage / execute code in a long-lived sandboxed IPython session.

    Args:
        action: ``"new_session"`` to spin one up, ``"execute"`` to run code,
            ``"close"`` to terminate, ``"list_sessions"`` to inspect.
        code: Required for ``execute`` (and optional for ``new_session``
            to run a setup snippet immediately).
        timeout: Per-call execution budget in seconds. Default 30.
        session_id: Required for ``execute`` / ``close``. Optional for
            ``new_session`` (auto-generated when omitted).
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "python_action",
            {
                "action": action,
                "code": code,
                "timeout": timeout,
                "session_id": session_id,
            },
        ),
    )
