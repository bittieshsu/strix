"""SDK function-tool wrapper for the legacy ``terminal_execute`` tool.

The terminal lives in the sandbox container — each persistent tmux
session is keyed by ``terminal_id`` on the in-container manager. The
host-side wrapper is a thin pass-through.
"""

from __future__ import annotations

import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._sandbox_dispatch import post_to_sandbox


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=180)
async def terminal_execute(
    ctx: RunContextWrapper,
    command: str,
    is_input: bool = False,
    timeout: float | None = None,
    terminal_id: str | None = None,
    no_enter: bool = False,
) -> str:
    """Run a shell command in the sandboxed Kali tmux session.

    Args:
        command: Shell command (or input for an interactive prompt when
            ``is_input=True``).
        is_input: Treat ``command`` as input to a running foreground process
            (e.g., feeding y/n to ``apt install``).
        timeout: Seconds to wait before returning partial output. Defaults
            to the in-container manager's policy.
        terminal_id: Persistent session selector. Defaults to ``"default"``.
        no_enter: When True, sends keystrokes without a trailing return.
            Useful for sending raw ANSI control sequences.
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "terminal_execute",
            {
                "command": command,
                "is_input": is_input,
                "timeout": timeout,
                "terminal_id": terminal_id,
                "no_enter": no_enter,
            },
        ),
    )
