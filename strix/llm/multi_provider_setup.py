"""Multi-provider routing setup.

Wraps the SDK's :class:`MultiProvider` and threads Strix's
``LLM_API_KEY`` / ``LLM_API_BASE`` into the underlying provider chain.

Routing:

- ``anthropic/<model>`` → :class:`AnthropicCachingLitellmModel` so
  prompt caching kicks in (we inject ``cache_control`` on the system
  message before the litellm call).
- ``openai/<model>`` (and bare model names) → SDK-native
  :class:`OpenAIProvider`, instantiated with our settings credentials so
  ``LLM_API_KEY`` works without forcing the user to also export
  ``OPENAI_API_KEY``. Keeps the Responses API as the default transport
  for genuine OpenAI usage.
- Every other prefix (``litellm/...``, ``any-llm/...``, …) falls through
  to whatever the SDK does natively.

Real-OpenAI vs OpenAI-compatible differentiation is by
``Settings.llm.api_base`` presence. If the user pointed at a non-default
base URL they're almost certainly on an OpenAI-compatible endpoint that
doesn't speak the Responses API, so we flip ``openai_use_responses=False``
to make the inner provider use chat-completions transport instead.
"""

from __future__ import annotations

import logging

from agents.exceptions import UserError
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider, MultiProviderMap

from strix.config import load_settings
from strix.llm.anthropic_cache_wrapper import AnthropicCachingLitellmModel


logger = logging.getLogger(__name__)


class _AnthropicCachingProvider(ModelProvider):
    """Routes ``anthropic/<model>`` aliases through
    :class:`AnthropicCachingLitellmModel`.

    The SDK's ``MultiProvider`` strips the matched prefix before calling
    ``get_model``, so we receive bare ``"<model>"`` (e.g.
    ``"claude-sonnet-4-6"``) and re-prefix with ``anthropic/`` so litellm
    routes to the Anthropic API.
    """

    def get_model(self, model_name: str | None) -> Model:
        if not model_name:
            raise UserError(
                "Anthropic provider requires a non-empty model name (e.g. 'claude-sonnet-4-6').",
            )
        full = model_name if model_name.startswith("anthropic/") else f"anthropic/{model_name}"
        logger.debug("Anthropic provider: building cached model for %s", full)
        return AnthropicCachingLitellmModel(model=full)


def build_multi_provider() -> MultiProvider:
    """Build the configured MultiProvider.

    Registers the ``anthropic/`` route through our caching wrapper and
    threads ``Settings.llm`` credentials into the SDK-native
    :class:`OpenAIProvider` so ``openai/<model>`` works with our single
    ``LLM_API_KEY`` env var. ``Settings.llm.api_base`` (when set) flips
    the OpenAI provider to chat-completions transport — the de-facto
    signal that the user is hitting an OpenAI-compatible endpoint that
    doesn't implement the Responses API.
    """
    pmap = MultiProviderMap()  # type: ignore[no-untyped-call]
    pmap.add_provider("anthropic", _AnthropicCachingProvider())

    llm = load_settings().llm
    use_responses = llm.api_base is None  # default endpoint → real OpenAI
    logger.debug(
        "MultiProvider built with anthropic/ cached + openai/ native "
        "(api_key=%s, base_url=%s, use_responses=%s)",
        "set" if llm.api_key else "unset",
        llm.api_base or "default",
        use_responses,
    )
    return MultiProvider(
        provider_map=pmap,
        openai_api_key=llm.api_key,
        openai_base_url=llm.api_base,
        openai_use_responses=use_responses,
    )
