"""Tool package.

The package init wires the in-container side: importing every tool
sub-package triggers the ``@register_tool`` decorations that populate
``strix.tools.registry.tools``, which the in-container FastAPI tool
server (:mod:`strix.runtime.tool_server`) dispatches against.

Host-side SDK function tools live in ``<family>/tool.py`` (or
``tools.py``) and are imported directly by
:mod:`strix.agents.factory` — they do not flow through this package
init's ``register_tool`` registry.
"""

from .agents_graph import *  # noqa: F403
from .browser import *  # noqa: F403
from .file_edit import *  # noqa: F403
from .finish import *  # noqa: F403
from .load_skill import *  # noqa: F403
from .notes import *  # noqa: F403
from .proxy import *  # noqa: F403
from .python import *  # noqa: F403
from .registry import (
    ImplementedInClientSideOnlyError,
    get_tool_by_name,
    get_tool_names,
    get_tools_prompt,
    needs_agent_state,
    register_tool,
    tools,
)
from .reporting import *  # noqa: F403
from .terminal import *  # noqa: F403
from .thinking import *  # noqa: F403
from .todo import *  # noqa: F403
from .web_search import *  # noqa: F403


__all__ = [
    "ImplementedInClientSideOnlyError",
    "get_tool_by_name",
    "get_tool_names",
    "get_tools_prompt",
    "needs_agent_state",
    "register_tool",
    "tools",
]
