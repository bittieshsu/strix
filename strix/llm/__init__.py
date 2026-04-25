"""LLM package — model provider, prompt-cache wrapper, session, dedup helper.

Side effects on import:

- Quiet litellm's debug logger (it spams ``logging.DEBUG`` on every
  request). The SDK's MultiProvider routes through litellm under the
  hood, and the debug stream pollutes the run-directory event log.
- Quiet asyncio's RuntimeWarning + drop its log propagation; some
  litellm async paths emit benign cleanup warnings.
"""

import logging
import warnings

import litellm


litellm._logging._disable_debugging()  # type: ignore[no-untyped-call]
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").propagate = False
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")
