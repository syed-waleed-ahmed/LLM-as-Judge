"""LLM provider implementations.

The judge is written against the :class:`~llm_judge.providers.base.LLMProvider`
protocol, so adding a new backend (OpenAI, Anthropic, a local model, …) is a
matter of implementing ``complete_json`` and does not touch the judging logic.
"""

from __future__ import annotations

from .base import ChatMessage, LLMProvider, ProviderHealth, UsageTracker
from .groq_provider import GroqProvider
from .mock_provider import MockProvider
from .ollama_provider import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider
from .quota import CallBudget, QuotaExceededError, QuotaProvider
from .registry import (
    ProviderInfo,
    build_provider,
    configuration_status,
    provider_info,
    provider_names,
    register_provider,
)

__all__ = [
    "ChatMessage",
    "LLMProvider",
    "ProviderHealth",
    "UsageTracker",
    "GroqProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "CallBudget",
    "QuotaExceededError",
    "QuotaProvider",
    "ProviderInfo",
    "build_provider",
    "configuration_status",
    "provider_info",
    "provider_names",
    "register_provider",
]
