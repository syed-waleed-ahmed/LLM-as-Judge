"""Lightweight, opt-in logging configuration.

Libraries should not configure the root logger on import, so we expose a helper
that applications (the CLI, the Streamlit app) call explicitly. Library modules
just use ``logging.getLogger(__name__)`` and emit records; if the host app never
configures logging, those records are silently dropped as Python intends.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure a simple console logger for the ``llm_judge`` namespace.

    Idempotent: calling it repeatedly will not attach duplicate handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger("llm_judge")
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        _CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
