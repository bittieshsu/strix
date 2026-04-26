"""Caido proxy tools — host-side ``@function_tool`` wrappers around ``_calls``.

The five tools delegate to :mod:`strix.tools.proxy._calls` for the actual
caido-sdk-client work and add LLM-friendly JSON serialization + error
wrapping on top. The shared call layer is also reused by the
``python_action`` tool to expose the same proxy surface inside the
sandbox's Python kernel — single source of truth for the SDK shapes.

Tools: ``list_requests``, ``view_request``, ``send_request``,
``repeat_request``, ``scope_rules``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import is_dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from agents import RunContextWrapper, function_tool

from strix.tools.proxy import _calls


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from caido_sdk_client import Client

    from strix.tools.proxy._calls import RequestPart, SortBy, SortOrder
else:
    # Runtime import: ``function_tool`` resolves the annotations via
    # ``typing.get_type_hints`` so the Literal aliases must be reachable
    # in module globals at decoration time even though they're "only"
    # used in annotations.
    from strix.tools.proxy._calls import (  # noqa: TC001
        RequestPart,
        SortBy,
        SortOrder,
    )


ScopeAction = Literal["get", "list", "create", "update", "delete"]


def _ctx_client(ctx: RunContextWrapper) -> Client | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    return inner.get("caido_client")


def _serialize(value: Any) -> Any:
    """Recursively convert SDK dataclasses/Pydantic objects to JSON-safe primitives."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return value.hex()
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_serialize(v) for v in value]
    return str(value)


def _no_client() -> str:
    return json.dumps(
        {"success": False, "error": "Caido client not available in run context"},
        ensure_ascii=False,
        default=str,
    )


def _err(name: str, exc: Exception) -> str:
    logger.exception("%s failed", name)
    return json.dumps(
        {"success": False, "error": f"{name} failed: {exc}"},
        ensure_ascii=False,
        default=str,
    )


# ----------------------------------------------------------------------
# list_requests
# ----------------------------------------------------------------------
@function_tool(timeout=120)
async def list_requests(
    ctx: RunContextWrapper,
    httpql_filter: str | None = None,
    first: int = 50,
    after: str | None = None,
    sort_by: SortBy = "timestamp",
    sort_order: SortOrder = "desc",
    scope_id: str | None = None,
) -> str:
    """List captured HTTP requests from the Caido proxy with HTTPQL filtering.

    Caido HTTPQL syntax (operators differ by field type):

    - **Integer fields** (``resp.code``, ``req.port``, ``id``,
      ``roundtrip``) — ``eq``, ``gt``, ``gte``, ``lt``, ``lte``, ``ne``.
      Examples: ``resp.code.eq:200``, ``resp.code.gte:400``,
      ``req.port.eq:443``.
    - **Text/byte fields** (``req.method``, ``req.host``, ``req.path``,
      ``req.query``, ``req.ext``, ``req.raw``) — ``regex``, ``cont``
      (substring), ``eq``. Examples: ``req.method.eq:"POST"``,
      ``req.path.cont:"/api/"``, ``req.host.regex:".*\\.example\\.com"``.
    - **Date fields** (``req.created_at``) — ``gt``, ``lt`` with ISO
      timestamps: ``req.created_at.gt:"2024-01-01T00:00:00Z"``.
    - **Combine** with ``AND`` / ``OR``: ``req.method.eq:"POST" AND
      resp.code.gte:400``.
    - **Special**: ``source:intercept`` (only intercepted requests),
      ``preset:"name"``.

    For sitemap-style tree traversal use HTTPQL filters: drill into a
    host with ``req.host.eq:"example.com"`` then narrow paths with
    ``req.path.cont:"/api/"``.

    Pagination is cursor-based. Pass the ``end_cursor`` from the
    ``page_info`` of one call as ``after`` to the next.

    Args:
        httpql_filter: Caido HTTPQL query (optional).
        first: Number of entries to return (default 50).
        after: Cursor from a previous response's ``page_info.end_cursor``.
        sort_by: One of ``timestamp`` / ``host`` / ``method`` / ``path``
            / ``status_code`` / ``response_time`` / ``response_size``
            / ``source``.
        sort_order: ``asc`` or ``desc``.
        scope_id: Restrict to a Caido scope (managed via ``scope_rules``).
    """
    client = _ctx_client(ctx)
    if client is None:
        return _no_client()

    try:
        connection = await _calls.list_requests(
            client,
            httpql_filter=httpql_filter,
            first=first,
            after=after,
            sort_by=sort_by,
            sort_order=sort_order,
            scope_id=scope_id,
        )

        entries = []
        for edge in connection.edges:
            req = edge.node.request
            resp = edge.node.response
            entries.append(
                {
                    "cursor": edge.cursor,
                    "request": {
                        "id": req.id,
                        "host": req.host,
                        "port": req.port,
                        "method": req.method,
                        "path": req.path,
                        "query": req.query,
                        "is_tls": req.is_tls,
                        "created_at": req.created_at.isoformat(),
                    },
                    "response": (
                        {
                            "id": resp.id,
                            "status_code": resp.status_code,
                            "length": resp.length,
                            "roundtrip_ms": resp.roundtrip_time,
                            "created_at": resp.created_at.isoformat(),
                        }
                        if resp is not None
                        else None
                    ),
                },
            )

        return json.dumps(
            {
                "success": True,
                "entries": entries,
                "page_info": {
                    "has_next_page": connection.page_info.has_next_page,
                    "has_previous_page": connection.page_info.has_previous_page,
                    "start_cursor": connection.page_info.start_cursor,
                    "end_cursor": connection.page_info.end_cursor,
                },
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        return _err("list_requests", exc)


# ----------------------------------------------------------------------
# view_request
# ----------------------------------------------------------------------
@function_tool(timeout=60)
async def view_request(
    ctx: RunContextWrapper,
    request_id: str,
    part: RequestPart = "request",
    search_pattern: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> str:
    """View a captured request or its response, optionally regex-searched.

    Two modes:

    - **With** ``search_pattern`` (compact regex hits) — returns up to 20
      matches with ``before`` / ``after`` context and position. Useful
      for hunting reflected input, leaked URLs, hidden parameters.
    - **Without** ``search_pattern`` (full content with line pagination)
      — returns the page of raw content plus ``has_more`` flag.

    Common search patterns:

    - API endpoints: ``/api/[a-zA-Z0-9._/-]+``
    - URLs: ``https?://[^\\s<>"']+``
    - Query parameters: ``[?&][a-zA-Z0-9_]+=([^&\\s<>"']+)``
    - Specific input reflection: search for the value you submitted.

    Args:
        request_id: Request ID from ``list_requests``.
        part: ``"request"`` or ``"response"``.
        search_pattern: Optional regex; switches the response shape to
            compact hits.
        page: 1-indexed page number (only when no ``search_pattern``).
        page_size: Lines per page.
    """
    client = _ctx_client(ctx)
    if client is None:
        return _no_client()

    try:
        result = await _calls.get_request(client, request_id, part=part)
        if result is None:
            return json.dumps(
                {"success": False, "error": f"Request {request_id} not found"},
                ensure_ascii=False,
                default=str,
            )

        raw_bytes = (
            result.request.raw
            if part == "request"
            else (result.response.raw if result.response is not None else None)
        )
        if raw_bytes is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"No raw {part} for {request_id}",
                },
                ensure_ascii=False,
                default=str,
            )
        content = raw_bytes.decode("utf-8", errors="replace")

        if search_pattern:
            return json.dumps(_regex_hits(content, search_pattern), ensure_ascii=False, default=str)

        return json.dumps(
            _paginate_lines(content, page=page, page_size=page_size),
            ensure_ascii=False,
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        return _err("view_request", exc)


def _regex_hits(content: str, pattern: str) -> dict[str, Any]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"success": False, "error": f"Invalid regex: {exc}"}

    hits = []
    for match in regex.finditer(content):
        start, end = match.span()
        before = content[max(0, start - 40) : start]
        after = content[end : end + 40]
        hits.append(
            {
                "match": match.group(0),
                "position": start,
                "before": before,
                "after": after,
            },
        )
        if len(hits) >= 20:
            break

    return {"success": True, "hits": hits, "total_hits": len(hits)}


def _paginate_lines(content: str, *, page: int, page_size: int) -> dict[str, Any]:
    lines = content.splitlines()
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return {
        "success": True,
        "content": "\n".join(lines[start:end]),
        "page": page,
        "page_size": page_size,
        "total_lines": len(lines),
        "has_more": end < len(lines),
    }


# ----------------------------------------------------------------------
# send_request
# ----------------------------------------------------------------------
@function_tool(timeout=120, strict_mode=False)
async def send_request(
    ctx: RunContextWrapper,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: int = 30,
) -> str:
    """Send an arbitrary HTTP request through the Caido proxy.

    Use this for one-off probes (test endpoints, reach external APIs).
    For modifying-and-replaying a request you've already captured, use
    ``repeat_request`` instead — it inherits the original headers /
    cookies / auth and only patches the fields you specify.

    Args:
        method: ``"GET"`` / ``"POST"`` / ``"PUT"`` / ``"DELETE"`` / etc.
        url: Full URL with protocol.
        headers: Optional header dict.
        body: Optional request body string.
        timeout: Per-request timeout in seconds (default 30).
    """
    del timeout  # The SDK applies its own timeout via the GraphQL settings.
    client = _ctx_client(ctx)
    if client is None:
        return _no_client()

    try:
        connection, raw = _calls.build_raw_request(
            method=method, url=url, headers=headers or {}, body=body
        )
        result = await _calls.replay_send_raw(client, raw=raw, connection=connection)
        return _format_replay_result(result)
    except Exception as exc:  # noqa: BLE001
        return _err("send_request", exc)


# ----------------------------------------------------------------------
# repeat_request
# ----------------------------------------------------------------------
@function_tool(timeout=120, strict_mode=False)
async def repeat_request(
    ctx: RunContextWrapper,
    request_id: str,
    modifications: dict[str, Any] | None = None,
) -> str:
    """Repeat a captured request, optionally patching individual fields.

    The standard pentesting workflow with this tool:

    1. ``agent-browser`` (via ``exec_command``) or live target traffic
       → request gets captured by Caido.
    2. ``list_requests`` → find the request ID you want to manipulate.
    3. ``repeat_request`` → send a modified version (auth-bypass test,
       payload injection, parameter tampering).

    Mirrors the manual "browse → capture → modify → test" flow used in
    real pentesting. Inherits everything from the original request
    (headers, cookies, auth, method, URL) and overlays only the fields
    you specify in ``modifications``.

    Args:
        request_id: ID of the original request (from ``list_requests``).
        modifications: Patch dict. Recognized keys:

            - ``url`` — replace the URL.
            - ``params`` — dict of query-string keys to add/update.
            - ``headers`` — dict of headers to add/update.
            - ``body`` — replace the body string entirely.
            - ``cookies`` — dict of cookies to add/update.
    """
    client = _ctx_client(ctx)
    if client is None:
        return _no_client()
    mods = modifications or {}

    try:
        result = await _calls.get_request(client, request_id, part="request")
        if result is None or result.request.raw is None:
            return json.dumps(
                {"success": False, "error": f"Request {request_id} not found"},
                ensure_ascii=False,
                default=str,
            )

        original = result.request
        raw_str = result.request.raw.decode("utf-8", errors="replace")
        components = _calls.parse_raw_request(raw_str)
        full_url = _calls.full_url_from_components(original, components, mods)
        modified = _calls.apply_modifications(components, mods, full_url)
        connection, raw = _calls.build_raw_request(
            method=modified["method"],
            url=modified["url"],
            headers=modified["headers"],
            body=modified["body"],
        )
        replay = await _calls.replay_send_raw(client, raw=raw, connection=connection)
        return _format_replay_result(replay)
    except Exception as exc:  # noqa: BLE001
        return _err("repeat_request", exc)


def _format_replay_result(replay: dict[str, Any]) -> str:
    response_raw = replay.get("response_raw")
    response: dict[str, Any] | None = None
    if response_raw is not None:
        response = {"raw": response_raw.decode("utf-8", errors="replace")}
    return json.dumps(
        {
            "success": replay["status"] == "DONE",
            "status": replay["status"],
            "error": replay["error"],
            "session_id": replay["session_id"],
            "elapsed_ms": replay["elapsed_ms"],
            "response": response,
        },
        ensure_ascii=False,
        default=str,
    )


# ----------------------------------------------------------------------
# scope_rules
# ----------------------------------------------------------------------
@function_tool(timeout=60)
async def scope_rules(
    ctx: RunContextWrapper,
    action: ScopeAction,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    scope_id: str | None = None,
    scope_name: str | None = None,
) -> str:
    """CRUD on Caido scope rules (allow/deny patterns).

    Scopes filter which traffic Caido tools see. Use them to focus on a
    target, exclude noisy assets (CDNs, static files), or define a
    bug-bounty allowlist.

    Pattern semantics:

    - Glob wildcards: ``*`` (any), ``?`` (single), ``[abc]`` (one of),
      ``[a-z]`` (range), ``[^abc]`` (none of).
    - **Empty allowlist = allow all domains.**
    - **Denylist always overrides allowlist.**

    Common denylist for noisy static assets:
    ``["*.gif", "*.jpg", "*.png", "*.css", "*.js", "*.ico", "*.svg",
    "*woff*", "*.ttf"]``.

    Each scope has a unique id usable as ``scope_id`` in
    ``list_requests``.

    Args:
        action:

            - ``list`` — return all scopes.
            - ``get`` — single scope by ``scope_id``.
            - ``create`` — needs ``scope_name``, optionally
              ``allowlist`` / ``denylist``.
            - ``update`` — needs ``scope_id`` + ``scope_name``;
              allowlist / denylist replace the previous values.
            - ``delete`` — needs ``scope_id``.

        allowlist: Domain patterns to include (e.g.
            ``["*.example.com", "api.test.com"]``).
        denylist: Patterns to exclude.
        scope_id: Required for ``get`` / ``update`` / ``delete``.
        scope_name: Required for ``create`` / ``update``.
    """
    client = _ctx_client(ctx)
    if client is None:
        return _no_client()

    try:
        if action == "list":
            scopes = await _calls.scope_list(client)
            return json.dumps(
                {"success": True, "scopes": [_serialize(s) for s in scopes]},
                ensure_ascii=False,
                default=str,
            )
        if action == "get":
            if not scope_id:
                return json.dumps(
                    {"success": False, "error": "scope_id required for get"},
                    ensure_ascii=False,
                    default=str,
                )
            scope = await _calls.scope_get(client, scope_id)
            return json.dumps(
                {"success": True, "scope": _serialize(scope)}, ensure_ascii=False, default=str
            )
        if action == "create":
            if not scope_name:
                return json.dumps(
                    {"success": False, "error": "scope_name required for create"},
                    ensure_ascii=False,
                    default=str,
                )
            scope = await _calls.scope_create(
                client, name=scope_name, allowlist=allowlist, denylist=denylist
            )
            return json.dumps(
                {"success": True, "scope": _serialize(scope)}, ensure_ascii=False, default=str
            )
        if action == "update":
            if not scope_id or not scope_name:
                return json.dumps(
                    {
                        "success": False,
                        "error": "scope_id and scope_name required for update",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            scope = await _calls.scope_update(
                client, scope_id, name=scope_name, allowlist=allowlist, denylist=denylist
            )
            return json.dumps(
                {"success": True, "scope": _serialize(scope)}, ensure_ascii=False, default=str
            )
        # action == "delete" — exhaustive Literal
        if not scope_id:
            return json.dumps(
                {"success": False, "error": "scope_id required for delete"},
                ensure_ascii=False,
                default=str,
            )
        await _calls.scope_delete(client, scope_id)
        return json.dumps({"success": True, "deleted": scope_id}, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        return _err("scope_rules", exc)
