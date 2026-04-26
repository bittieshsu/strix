# Agent Orchestration Refactor

## Goal

Keep every Strix agent first-class and addressable. Root and subagents are all SDK agents with
their own `SQLiteSession`; any registered agent can be messaged, stopped, resumed, and observed.
Strix should not collapse to a manager-only pattern and should not rebuild the SDK model/tool loop.

## SDK Boundary

- The OpenAI Agents SDK owns model/tool execution through `Runner.run_streamed`.
- The SDK session owns per-agent conversation history and message replay.
- Strix owns only product semantics the SDK does not provide: agent ids, parent/child graph,
  wake/stop signals, TUI-visible status, snapshots, and process-resume metadata.
- SDK stream events are enough for tool/usage telemetry; lifecycle should not depend on custom
  `RunHooks`.

## Current File Shape

- `strix/orchestration/coordinator.py`: graph state, runtime handles, SDK session messaging,
  wake/stop signals, snapshots, resume, stream telemetry, and the interactive continuation loop.
- `strix/orchestration/scan.py`: top-level scan assembly only: sandbox setup, root construction,
  resume restore, session opening, and subagent respawn.
- `strix/tools/agents_graph/tools.py`: thin tool facade over `AgentCoordinator`.
- `strix/tools/finish/tool.py`: report finalization plus active-agent guard.

Removed custom orchestration modules:

- `bus.py`
- `filter.py`
- `hooks.py`
- `run_loop.py`

`agents.json` is the only graph/status snapshot. `agents.db` is the only SDK session database;
each agent uses its own SDK `session_id` inside that shared database.

## Lifecycle Semantics

- `running`: an SDK run cycle is active.
- `waiting`: parked and addressable.
- `completed`: task completed, parked, and still addressable in interactive mode.
- `llm_failed`: parked after SDK/model failure; only user input resumes it.
- `stopped`: gracefully stopped, parked, and addressable.
- `crashed` / `failed`: parked after failure and addressable for user recovery.

Unknown agent id is the only invalid message target. There is no routing-closed status.

## Tool Semantics

- `create_agent` registers a child, opens its SDK session, and starts its SDK runner task.
- `send_message_to_agent` appends a user-role item to the target SDK session and wakes the target.
- `wait_for_message` parks immediately in interactive mode; in non-interactive mode it blocks on
  the coordinator wake event and returns newly appended session items to the current run.
- `agent_finish` sends the parent completion report and returns a final-output marker; the
  coordinator settles status from that marker.
- `finish_scan` refuses to finalize while active agents remain and lets the coordinator settle root
  status from the successful final-output marker.
- `stop_agent` uses SDK streaming cancellation when a run is active and wakes parked agents so the
  continuation loop observes the stop.

## Remaining Regression Checks To Add

- Completed child remains messageable and resumes from its SDK session.
- Stopped/crashed/failed child remains messageable and resumes from user input.
- Interactive `wait_for_message` parks and returns control to the continuation loop.
- Non-interactive `wait_for_message` returns the newly appended session message content.
- `finish_scan` blocks while child agents are active.
- Resume rebuilds parked child runners from `agents.json` plus the shared `agents.db`.
- Graceful stop works for both active-stream and parked agents.
