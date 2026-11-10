"""Provider registry: build the configured backend by name, and discover plugins.

Built-in providers:

========================  ========  ==========  =======================================
Name                      API key   Runs where  Notes
========================  ========  ==========  =======================================
``groq`` (default)        yes       cloud       Free plan, rate-limited per model
``ollama``                no        local       Free; your hardware, your models
``openai_compatible``     optional  either      Any ``/chat/completions`` server
``mock``                  no        offline     Heuristic, *not an LLM*; demos/tests
========================  ========  ==========  =======================================

Third-party packages can add providers through the ``llm_judge.providers``
entry-point group. Each entry point must load to a zero-argument callable that
returns ``(ProviderInfo, factory)``, where ``factory(settings)`` builds the
provider.

The registry never switches providers or models behind the caller's back.
Whatever is configured is what runs, or the call fails loudly.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import LLMProvider

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[Settings], LLMProvider]
ENTRY_POINT_GROUP = "llm_judge.providers"


@dataclass(frozen=True)
class ProviderInfo:
    """Static description of a provider, shown by ``llm-judge providers``."""

    name: str
    description: str
    requires_api_key: bool
    local: bool
    cost: str


def _build_groq(settings: Settings) -> LLMProvider:
    from .groq_provider import GroqProvider

    return GroqProvider(
        api_key=settings.require_api_key(),
        model=settings.groq_model,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
    )


def _build_ollama(settings: Settings) -> LLMProvider:
    from .ollama_provider import OllamaProvider

    return OllamaProvider(
        settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        timeout=max(settings.request_timeout, 120.0),
    )


def _build_openai_compatible(settings: Settings) -> LLMProvider:
    from .openai_compatible import OpenAICompatibleProvider

    if not settings.openai_base_url or not settings.openai_model:
        raise ConfigurationError(
            "The openai_compatible provider needs JUDGE_OPENAI_BASE_URL and JUDGE_OPENAI_MODEL."
        )
    return OpenAICompatibleProvider(
        settings.openai_base_url,
        settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        json_mode=settings.openai_json_mode,
    )


def _build_mock(settings: Settings) -> LLMProvider:
    from .mock_provider import MockProvider

    return MockProvider()


_lock = threading.Lock()
_plugins_loaded = False
_REGISTRY: dict[str, tuple[ProviderInfo, ProviderFactory]] = {
    "groq": (
        ProviderInfo(
            "groq",
            "Groq-hosted open models (default judge: openai/gpt-oss-120b).",
            requires_api_key=True,
            local=False,
            cost="free plan, rate-limited",
        ),
        _build_groq,
    ),
    "ollama": (
        ProviderInfo(
            "ollama",
            "Local models served by Ollama on your own machine.",
            requires_api_key=False,
            local=True,
            cost="free (your hardware)",
        ),
        _build_ollama,
    ),
    "openai_compatible": (
        ProviderInfo(
            "openai_compatible",
            "Any OpenAI-compatible /chat/completions endpoint (LM Studio, vLLM, ...).",
            requires_api_key=False,
            local=False,
            cost="depends on the endpoint",
        ),
        _build_openai_compatible,
    ),
    "mock": (
        ProviderInfo(
            "mock",
            "Deterministic heuristic judge - NOT an LLM. For demos, tests and keyless use.",
            requires_api_key=False,
            local=True,
            cost="free (offline)",
        ),
        _build_mock,
    ),
}


def register_provider(
    info: ProviderInfo, factory: ProviderFactory, *, replace: bool = False
) -> None:
    """Register a provider under ``info.name``. Refuses to overwrite unless ``replace``."""
    with _lock:
        if info.name in _REGISTRY and not replace:
            raise ConfigurationError(f"A provider named '{info.name}' is already registered.")
        _REGISTRY[info.name] = (info, factory)


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            info, factory = ep.load()()
            register_provider(info, factory)
        except Exception as exc:  # a broken plugin must not break the built-ins
            logger.warning("Skipping provider plugin '%s': %s", ep.name, exc)


def provider_names() -> list[str]:
    """Names of all registered providers (built-ins first, then plugins)."""
    _load_plugins()
    return list(_REGISTRY)


def provider_info(name: str) -> ProviderInfo:
    _load_plugins()
    try:
        return _REGISTRY[name][0]
    except KeyError:
        raise ConfigurationError(
            f"Unknown provider '{name}'. Available: {', '.join(provider_names())}."
        ) from None


def configuration_status(name: str, settings: Settings) -> tuple[bool, str]:
    """Whether ``name`` has the settings it needs, with a short explanation.

    This only checks configuration. Use a provider's ``health()`` to check
    that it is actually reachable.
    """
    provider_info(name)  # validates the name
    if name == "groq":
        if settings.groq_api_key:
            return True, f"GROQ_API_KEY set; model {settings.groq_model}"
        return False, "set GROQ_API_KEY (free key: https://console.groq.com/keys)"
    if name == "ollama":
        return True, f"{settings.ollama_base_url}, model {settings.ollama_model}"
    if name == "openai_compatible":
        if settings.openai_base_url and settings.openai_model:
            return True, f"{settings.openai_base_url}, model {settings.openai_model}"
        return False, "set JUDGE_OPENAI_BASE_URL and JUDGE_OPENAI_MODEL"
    if name == "mock":
        return True, "always available (heuristic, not an LLM)"
    return True, "plugin provider"


def fallback_names(settings: Settings) -> list[str]:
    """The configured fallback providers (``JUDGE_FALLBACK_PROVIDERS``), validated."""
    names = [n.strip() for n in settings.fallback_providers.split(",") if n.strip()]
    if not names:
        return []
    if len(set(names)) != len(names):
        raise ConfigurationError(f"JUDGE_FALLBACK_PROVIDERS lists a provider twice: {names}")
    if settings.provider in names:
        raise ConfigurationError(
            f"JUDGE_FALLBACK_PROVIDERS can't include the primary provider '{settings.provider}'."
        )
    if "mock" in names:
        raise ConfigurationError(
            "The mock judge can't be a fallback: its scores are not LLM judgments and must "
            "not be mixed into live results. Choose another provider, or use "
            "JUDGE_PROVIDER=mock explicitly."
        )
    for fallback in names:
        provider_info(fallback)  # unknown names raise a helpful error
    return names


def build_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    """Build the provider named ``name`` (default: ``settings.provider``).

    When ``JUDGE_FALLBACK_PROVIDERS`` is set and the configured primary provider
    is being built, the result is a
    :class:`~llm_judge.providers.fallback.FallbackProvider` that tries the
    fallbacks in order when the primary fails.
    """
    chosen = name or settings.provider
    provider_info(chosen)  # raises a helpful ConfigurationError for unknown names
    primary = _REGISTRY[chosen][1](settings)
    if chosen != settings.provider:
        return primary
    names = fallback_names(settings)
    if not names:
        return primary
    from .fallback import FallbackProvider

    return FallbackProvider(primary, [_REGISTRY[n][1](settings) for n in names])
