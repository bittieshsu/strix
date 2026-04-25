"""SDK function-tool wrapper for the legacy ``finish_scan`` tool.

The legacy function:

- Validates the caller is the root agent (``parent_id is None``).
- Checks no other agents are still running (via the legacy
  ``_agent_graph`` global).
- Persists the four executive-summary fields via
  ``get_global_tracer().update_scan_final_fields(...)``.
- Reports the final vulnerability count.

Both the parent-id check and the agent-graph check rely on legacy
multi-agent state that Phase 3 will reimplement on top of the SDK
``RunContextWrapper`` + a per-run registry. Until Phase 3 lands, the
legacy adapter returns an object with no ``parent_id`` attribute —
``hasattr`` returns False, the validation skips, and the call proceeds
as if invoked by a root agent. That's the correct degenerate behavior
in single-agent mode, which is all Phase 2 ships.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._state_adapter import adapter_from_ctx
from strix.tools.finish import finish_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=60)
async def finish_scan(
    ctx: RunContextWrapper,
    executive_summary: str,
    methodology: str,
    technical_analysis: str,
    recommendations: str,
) -> str:
    """Finalize the scan and persist the four executive summary sections.

    Only the root agent should call this. Subagents should use
    ``agent_finish`` from the agents_graph tool family instead.

    Args:
        executive_summary: High-level scan outcome.
        methodology: Approach taken.
        technical_analysis: Findings detail across the engagement.
        recommendations: Prioritized fix list.
    """
    state = adapter_from_ctx(ctx)
    return _dump(
        await asyncio.to_thread(
            _impl.finish_scan,
            executive_summary=executive_summary,
            methodology=methodology,
            technical_analysis=technical_analysis,
            recommendations=recommendations,
            agent_state=state,
        ),
    )
