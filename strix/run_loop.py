"""``run_with_continuation`` — interactive-mode demo-loop wrapper around ``Runner.run``.

Pre-migration ``BaseAgent.agent_loop`` ran forever in interactive mode,
re-entering a "waiting state" after each finish-tool call so the agent
could pick up follow-up messages from its parent (or from the user, in
the root's case). Post-migration ``Runner.run`` returns on
``StopAtTools(...)`` and the agent is gone.

This helper restores the legacy semantics using the SDK's canonical
demo-loop pattern (``agents/repl.py:run_demo_loop``): after each
``Runner.run`` cycle, ``await bus.wait_for_message(agent_id)``, drain
new messages, and re-invoke ``Runner.run`` with them as the next turn's
input. The session (if provided) preserves prior conversation across
cycles.

Used by both the root scan loop in ``entry.run_strix_scan`` and the
child-agent loop in ``tools.agents_graph.tools.create_agent`` so every
interactive agent on the bus stays alive.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from agents import Runner


if TYPE_CHECKING:
    from agents.lifecycle import RunHooks
    from agents.memory import Session
    from agents.result import RunResult
    from agents.run_config import RunConfig

    from strix.orchestration.bus import AgentMessageBus


logger = logging.getLogger(__name__)


async def run_with_continuation(
    *,
    agent: Any,
    initial_input: Any,
    run_config: RunConfig,
    context: dict[str, Any],
    hooks: RunHooks[Any],
    max_turns: int,
    bus: AgentMessageBus,
    agent_id: str,
    interactive: bool,
    session: Session | None = None,
) -> RunResult:
    """Run an agent once (non-interactive) or in a continuation loop (interactive).

    For non-interactive runs this is a thin wrapper around
    ``Runner.run`` and returns its result.

    For interactive runs the function loops: after each ``Runner.run``
    returns, it awaits ``bus.wait_for_message(agent_id)``, drains any
    accumulated messages from the inbox, formats them as the next
    turn's user input, and invokes ``Runner.run`` again. The loop ends
    when the wait gets cancelled (e.g. parent ``cancel_descendants`` or
    user-issued KeyboardInterrupt).
    """
    kwargs: dict[str, Any] = {
        "input": initial_input,
        "run_config": run_config,
        "context": context,
        "hooks": hooks,
        "max_turns": max_turns,
    }
    if session is not None:
        kwargs["session"] = session

    result: RunResult = await Runner.run(agent, **kwargs)

    if not interactive:
        return result

    while True:
        try:
            await bus.wait_for_message(agent_id)
        except asyncio.CancelledError:
            return result

        pending = await bus.drain(agent_id)
        if not pending:
            continue
        next_input = "\n\n".join(
            str(msg.get("content", "")).strip() for msg in pending if msg.get("content")
        )
        if not next_input:
            continue

        kwargs["input"] = next_input
        result = await Runner.run(agent, **kwargs)
