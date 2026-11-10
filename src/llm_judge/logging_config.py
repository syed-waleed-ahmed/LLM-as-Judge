"""Lightweight, opt-in logging configuration.

Libraries should not configure the root logger on import, so we expose a helper
that applications (the CLI, the Streamlit app) call explicitly. Library modules
just use ``logging.getLogger(__name__)`` and emit records; if the host app never
configures logging, those records are silently dropped as Python intends.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, TextIO

_CONFIGURED = False


if TYPE_CHECKING:
    _Handler = logging.StreamHandler[TextIO]
else:  # StreamHandler isn't subscriptable at runtime before Python 3.11
    _Handler = logging.StreamHandler


class _StderrHandler(_Handler):
    """Writes to whatever ``sys.stderr`` is at the time of each record.

    A plain ``StreamHandler`` keeps the stream it was created with. If that
    stream is later replaced and closed (as test runners and some hosts do),
    every log call would fail with "I/O operation on closed file".
    """

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass  # always follow sys.stderr


def configure_logging(level: str = "INFO") -> None:
    """Configure a simple console logger for the ``llm_judge`` namespace.

    Idempotent: calling it repeatedly will not attach duplicate handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger("llm_judge")
    if not _CONFIGURED:
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
        logger.addHandler(handler)
        _CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
