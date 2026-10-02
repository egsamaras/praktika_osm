"""Exception hierarchy for Praktika.

Every domain error derives from ``PraktikaError`` so callers can catch one type at a CLI or API
boundary. None of these derive from ``ValueError``: Pydantic wraps ``ValueError`` raised inside
validators into ``ValidationError``, and the egress check must surface as ``EgressError`` unchanged.
"""

from __future__ import annotations


class PraktikaError(Exception):
    """Base class for all Praktika errors."""


class ConfigError(PraktikaError):
    """The configuration cannot be loaded, for example ``PRAKTIKA_ENV_FILE`` names a file that
    does not exist or cannot be read."""


class EgressError(PraktikaError):
    """A configured or requested URL targets a host outside the egress allow-list (C-01)."""


class ConsentRefused(PraktikaError):  # noqa: N818 — established public name
    """The consent gate refused to proceed (not notified, objections, or failed checks) (C-03)."""


class ScopeError(PraktikaError):
    """The meeting falls under a pilot scope exclusion (C-10)."""


class ClassificationNotAllowed(PraktikaError):  # noqa: N818 — established public name
    """The classification is not permitted in the current mode or pilot state (C-09)."""


class LanguageRefused(PraktikaError):  # noqa: N818 — named like the other refusals
    """The requested language mode is switched off in this deployment (``--lang ar-mixed``
    while ``stt_ar = none``). Raised before the consent script is shown or any audio is read;
    the CLI exits 2 (refusal), like the other refusals."""


class LLMError(PraktikaError):
    """Transport or protocol failure talking to the LLM backend."""


class LLMSchemaError(LLMError):
    """The LLM returned output that failed schema validation after the permitted retry."""


class NotApproved(PraktikaError):  # noqa: N818 — established public name
    """An operation requires approved minutes but the minutes are not approved (C-07)."""


class ModelRegisterMismatch(PraktikaError):  # noqa: N818 — established public name
    """Local model weights do not match the hashes recorded in ``models.yaml`` (C-13)."""


class FfmpegError(PraktikaError):
    """ffmpeg is missing or exited with an error; the message carries its stderr."""
