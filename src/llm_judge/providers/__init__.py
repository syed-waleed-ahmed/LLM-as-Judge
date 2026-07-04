"""LLM provider implementations.

The judge is written against the :class:`~llm_judge.providers.base.LLMProvider`
protocol, so adding a new backend (OpenAI, Anthropic, a local model, …) is a
matter of implementing ``complete_json`` and does not touch the judging logic.
"""

from __future__ import annotations

from .base import ChatMessage, LLMProvider
from .groq_provider import GroqProvider

__all__ = ["ChatMessage", "LLMProvider", "GroqProvider"]
