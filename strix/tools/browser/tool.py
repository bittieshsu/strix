"""SDK function-tool wrapper for the legacy ``browser_action`` tool.

The browser is fully sandbox-bound — the legacy implementation runs
inside the container against a Playwright instance the tool server
manages. We delegate every action verbatim to ``post_to_sandbox``.

The legacy ``browser_action`` is a single mega-tool dispatching 21
discrete actions (launch, goto, click, scroll_*, new_tab, etc.). We
preserve that shape for parity rather than fanning out into 21
separate tools — that would balloon the system prompt and surprise
the model.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._sandbox_dispatch import post_to_sandbox


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


BrowserAction = Literal[
    "launch",
    "goto",
    "click",
    "type",
    "scroll_down",
    "scroll_up",
    "back",
    "forward",
    "new_tab",
    "switch_tab",
    "close_tab",
    "wait",
    "execute_js",
    "double_click",
    "hover",
    "press_key",
    "save_pdf",
    "get_console_logs",
    "view_source",
    "close",
    "list_tabs",
]


# Browser actions can take time (page loads, navigation timeouts), so
# match the sandbox dispatch read budget rather than capping shorter.
@strix_tool(timeout=180)
async def browser_action(
    ctx: RunContextWrapper,
    action: BrowserAction,
    url: str | None = None,
    coordinate: str | None = None,
    text: str | None = None,
    tab_id: str | None = None,
    js_code: str | None = None,
    duration: float | None = None,
    key: str | None = None,
    file_path: str | None = None,
    clear: bool = False,
) -> str:
    """Drive the sandboxed Playwright browser.

    Args:
        action: The browser action to dispatch — see ``BrowserAction``
            literal for the full set.
        url: Required for ``launch`` / ``goto`` / ``new_tab`` (with URL).
        coordinate: ``"x,y"`` pixel target for click/hover/double_click.
        text: Required for ``type``.
        tab_id: Optional explicit tab targeting; defaults to the active tab.
        js_code: Required for ``execute_js``.
        duration: Seconds to wait for ``wait`` action.
        key: Required for ``press_key`` (e.g. ``"Enter"``, ``"Escape"``).
        file_path: Required for ``save_pdf``.
        clear: For ``type``, clears the field first.
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "browser_action",
            {
                "action": action,
                "url": url,
                "coordinate": coordinate,
                "text": text,
                "tab_id": tab_id,
                "js_code": js_code,
                "duration": duration,
                "key": key,
                "file_path": file_path,
                "clear": clear,
            },
        ),
    )
