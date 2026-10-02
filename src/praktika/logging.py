"""Structured logging via structlog.

Contract: ``configure_logging`` is idempotent and safe to call from the CLI and the server; every
logger returned by ``get_logger`` emits ISO-8601 UTC timestamps, the level, the logger name and any
values bound with ``bind_request_id`` / ``structlog.contextvars``. Nothing in this package uses
``print`` (ruff T20); operator-facing output goes through ``rich`` in the CLI and through this
module everywhere else.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any

import structlog

_CONFIGURED = False
NO_COLOR_VAR = "NO_COLOR"


def stderr_wants_colour() -> bool:
    """True only when stderr is an interactive terminal and ``NO_COLOR`` is unset or empty.

    Colour codes are for a person at a terminal. Redirected to a file, piped, captured by the
    journal or by a test harness, stderr gets plain text, as the ``NO_COLOR`` convention asks. A
    stderr that cannot answer (closed, or replaced by an object without ``isatty``) counts as
    not a terminal.
    """
    if os.environ.get(NO_COLOR_VAR):
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def configure_logging(json: bool = False, level: str = "INFO") -> None:
    """Configure stdlib logging and structlog once.

    ``json=True`` renders one JSON object per line (service mode, Fluent Bit / SIEM friendly);
    ``json=False`` renders a console line for local use, coloured only when stderr is a terminal
    (see ``stderr_wants_colour``), so a pipe, a file, the journal or captured test evidence never
    carries escape sequences. ``level`` is a stdlib level name such as ``"INFO"`` or ``"DEBUG"``;
    an unknown name raises ``ValueError``.
    """
    global _CONFIGURED
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {level!r}")

    renderer: structlog.types.Processor
    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=stderr_wants_colour())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=numeric, stream=sys.stderr, format="%(message)s", force=True)
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger named ``name``; configures console logging if nothing has yet."""
    if not _CONFIGURED:
        configure_logging(json=False, level="INFO")
    return structlog.get_logger(name)


def bind_request_id(request_id: str | None = None) -> str:
    """Bind a request id into the logging context for the current task and return it.

    A fresh UUID4 is generated when ``request_id`` is ``None``. Call ``clear_request_id`` at the
    end of the request.
    """
    rid = request_id or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=rid)
    return rid


def clear_request_id() -> None:
    """Remove any bound request id (and other context values) from the logging context."""
    structlog.contextvars.clear_contextvars()


def bind_context(**values: Any) -> None:
    """Bind arbitrary key/value pairs (for example ``meeting_id``) into the logging context."""
    structlog.contextvars.bind_contextvars(**values)
