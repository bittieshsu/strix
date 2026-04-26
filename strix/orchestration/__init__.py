"""Strix multi-agent orchestration on top of OpenAI Agents SDK.

- :class:`AgentCoordinator` owns Strix-specific graph/status/wake state.
- SDK ``SQLiteSession`` owns per-agent conversation history and message
  transport.
- ``runner.py`` owns SDK ``Runner.run_streamed`` and child-agent spawning.

Import deeply (for example, ``from strix.orchestration.coordinator
import AgentCoordinator``) so ``import strix.orchestration`` doesn't
drag every submodule's deps in eagerly.
"""
