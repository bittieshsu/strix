"""SDK function-tool wrapper for the legacy ``load_skill`` tool.

The legacy implementation reaches into ``_agent_instances`` (a global
dict the legacy multi-agent orchestrator maintains) to find the running
``Agent`` instance and call ``agent.llm.add_skills(...)``. That global
goes away under the SDK migration — Phase 3 will replace it with a
context-keyed registry, and this wrapper will be updated to read from
that registry.

For Phase 2 we ship the wrapper as-is. The legacy function falls back
to a clean error path when the agent instance lookup fails, so the
tool degrades gracefully ("Could not find running agent instance...")
until Phase 3 lands. That's better than crashing or stubbing out the
tool entirely — the model still gets a structured error it can react to.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools._state_adapter import adapter_from_ctx
from strix.tools.load_skill import load_skill_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


@strix_tool(timeout=60)
async def load_skill(ctx: RunContextWrapper, skills: str) -> str:
    """Load one or more named skills into this agent's prompt context.

    Args:
        skills: Comma-separated skill names (max 5). E.g.
            ``"recon,xss,sqli"``. Skill discovery uses
            ``strix.skills.parse_skill_list``.
    """
    state = adapter_from_ctx(ctx)
    return _dump(
        await asyncio.to_thread(_impl.load_skill, agent_state=state, skills=skills),
    )
