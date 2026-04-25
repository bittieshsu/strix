"""SDK function-tool wrappers for the seven Caido proxy tools.

All seven dispatch to the in-container Caido manager via the sandbox
tool server. Same pattern as browser/terminal/python — host wrapper is
pure pass-through, no logic of its own.

Tools: list_requests, view_request, send_request, repeat_request,
scope_rules, list_sitemap, view_sitemap_entry.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._sandbox_dispatch import post_to_sandbox


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


RequestPart = Literal["request", "response"]
SortBy = Literal[
    "timestamp",
    "host",
    "method",
    "path",
    "status_code",
    "response_time",
    "response_size",
    "source",
]
SortOrder = Literal["asc", "desc"]
SitemapDepth = Literal["DIRECT", "ALL"]
ScopeAction = Literal["get", "list", "create", "update", "delete"]


@strix_tool(timeout=120)
async def list_requests(
    ctx: RunContextWrapper,
    httpql_filter: str | None = None,
    start_page: int = 1,
    end_page: int = 1,
    page_size: int = 50,
    sort_by: SortBy = "timestamp",
    sort_order: SortOrder = "desc",
    scope_id: str | None = None,
) -> str:
    """List captured HTTP requests from the Caido proxy.

    Args:
        httpql_filter: Caido HTTPQL query (e.g. ``"resp.code:eq:500"``).
        start_page / end_page: Inclusive page range to return.
        page_size: Entries per page; default 50.
        sort_by: Field to sort by.
        sort_order: ``"asc"`` or ``"desc"``.
        scope_id: Restrict to a specific scope.
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "list_requests",
            {
                "httpql_filter": httpql_filter,
                "start_page": start_page,
                "end_page": end_page,
                "page_size": page_size,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "scope_id": scope_id,
            },
        ),
    )


@strix_tool(timeout=60)
async def view_request(
    ctx: RunContextWrapper,
    request_id: str,
    part: RequestPart = "request",
    search_pattern: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> str:
    """View a single captured request or its response, with optional regex highlight."""
    return _dump(
        await post_to_sandbox(
            ctx,
            "view_request",
            {
                "request_id": request_id,
                "part": part,
                "search_pattern": search_pattern,
                "page": page,
                "page_size": page_size,
            },
        ),
    )


# strict_mode=False because ``headers`` is a free-form dict — the model
# can't enumerate all possible HTTP headers, and the SDK's strict JSON
# schema rejects ``additionalProperties: true``.
@strix_tool(timeout=120, strict_mode=False)
async def send_request(
    ctx: RunContextWrapper,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: int = 30,
) -> str:
    """Send an arbitrary HTTP request through the Caido proxy.

    Args:
        method: ``"GET"``, ``"POST"``, etc.
        url: Full URL.
        headers: Optional header dict.
        body: Optional body string.
        timeout: Per-request timeout in seconds.
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "send_request",
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "body": body,
                "timeout": timeout,
            },
        ),
    )


# strict_mode=False because ``modifications`` is a free-form patch dict
# (header overrides, body replacements, query-string tweaks) the model
# composes per-call.
@strix_tool(timeout=120, strict_mode=False)
async def repeat_request(
    ctx: RunContextWrapper,
    request_id: str,
    modifications: dict[str, Any] | None = None,
) -> str:
    """Repeat a captured request, optionally applying field modifications."""
    return _dump(
        await post_to_sandbox(
            ctx,
            "repeat_request",
            {
                "request_id": request_id,
                "modifications": modifications or {},
            },
        ),
    )


@strix_tool(timeout=60)
async def scope_rules(
    ctx: RunContextWrapper,
    action: ScopeAction,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    scope_id: str | None = None,
    scope_name: str | None = None,
) -> str:
    """CRUD on Caido scope rules (allow/deny lists)."""
    return _dump(
        await post_to_sandbox(
            ctx,
            "scope_rules",
            {
                "action": action,
                "allowlist": allowlist,
                "denylist": denylist,
                "scope_id": scope_id,
                "scope_name": scope_name,
            },
        ),
    )


@strix_tool(timeout=60)
async def list_sitemap(
    ctx: RunContextWrapper,
    scope_id: str | None = None,
    parent_id: str | None = None,
    depth: SitemapDepth = "DIRECT",
    page: int = 1,
) -> str:
    """List Caido sitemap entries (proxied URL tree).

    Args:
        scope_id: Restrict to a scope.
        parent_id: Drill into a specific subtree.
        depth: ``"DIRECT"`` (direct children only) or ``"ALL"`` (recursive).
        page: 1-indexed page number.
    """
    return _dump(
        await post_to_sandbox(
            ctx,
            "list_sitemap",
            {
                "scope_id": scope_id,
                "parent_id": parent_id,
                "depth": depth,
                "page": page,
            },
        ),
    )


@strix_tool(timeout=60)
async def view_sitemap_entry(ctx: RunContextWrapper, entry_id: str) -> str:
    """Fetch a single sitemap entry's metadata + linked requests."""
    return _dump(
        await post_to_sandbox(ctx, "view_sitemap_entry", {"entry_id": entry_id}),
    )
