"""Tool package.

Importing every sub-package triggers the ``@register_tool``
decorations that populate ``strix.tools.registry.tools``. The
in-container FastAPI tool server (:mod:`strix.runtime.tool_server`)
dispatches against that registry.

Host-side SDK function tools live in ``<family>/tool[s].py`` and are
imported directly by :mod:`strix.agents.factory` — they don't flow
through this registry.
"""

from .agents_graph import *  # noqa: F403
from .browser import *  # noqa: F403
from .file_edit import *  # noqa: F403
from .finish import *  # noqa: F403
from .notes import *  # noqa: F403
from .proxy import *  # noqa: F403
from .python import *  # noqa: F403
from .registry import get_tool_by_name, get_tool_names, register_tool, tools
from .reporting import *  # noqa: F403
from .terminal import *  # noqa: F403
from .thinking import *  # noqa: F403
from .todo import *  # noqa: F403
from .web_search import *  # noqa: F403


__all__ = [
    "get_tool_by_name",
    "get_tool_names",
    "register_tool",
    "tools",
]
