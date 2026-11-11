"""REST API for LLM-as-Judge (FastAPI). Install with ``pip install "llm-as-judge-platform[api]"``.

Start a local server with ``llm-judge serve`` and open http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

try:
    from .app import create_app
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "The REST API needs FastAPI and Uvicorn: pip install 'llm-as-judge-platform[api]'"
    ) from exc

__all__ = ["create_app"]
