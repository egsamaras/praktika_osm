"""Policy decisions (C-07, C-09): the single hook where an organisation's policy engine plugs in.

Contract: every function returns a ``PolicyDecision`` and never raises; callers decide whether
to raise. ``evaluate`` is default-deny: an unknown action is refused. The rules encode the
classification permitted per deployment mode (C-09) and control C-07 (no export before
approval; de-tokenisation only in approved, non-restricted minutes); docs/CONTROLS.md lists
both.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from praktika.config import Settings
from praktika.models import Classification, Minutes


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str
    obligations: list[str] = []


def _classification(value: Classification | str) -> Classification | None:
    try:
        return Classification(value)
    except ValueError:
        return None


def classification_allowed(
    classification: Classification | str, settings: Settings
) -> PolicyDecision:
    """Is this classification permitted in the current deployment?

    Pilot (``settings.pilot``): ``internal`` only. Local mode: never ``restricted`` and never
    ``confidential`` (both need the on-prem service). Service mode outside the pilot allows
    ``confidential``; ``restricted`` additionally carries the MNPI-mode obligations.
    """
    cls = _classification(classification)
    if cls is None:
        return PolicyDecision(allowed=False, reason=f"unknown classification {classification!r}")
    if settings.pilot and cls is not Classification.internal:
        return PolicyDecision(
            allowed=False, reason=f"pilot accepts internal meetings only, not {cls.value}"
        )
    if settings.mode == "local" and cls is not Classification.internal:
        return PolicyDecision(
            allowed=False, reason=f"{cls.value} requires PRAKTIKA_MODE=service, not local"
        )
    if cls is Classification.restricted:
        return PolicyDecision(
            allowed=True,
            reason="restricted permitted in service mode outside the pilot (MNPI mode)",
            obligations=["no_fts_index", "no_audio_on_disk", "watermark_exports"],
        )
    return PolicyDecision(allowed=True, reason=f"{cls.value} permitted")


def _is_approved(minutes: Minutes) -> bool:
    return minutes.review.status == "approved" and not minutes.blocking_flags()


def export_allowed(
    minutes: Minutes, settings: Settings | None = None, *, allow_draft: bool = False
) -> PolicyDecision:
    """Clean export needs approved minutes with no open priority-1 flag (C-07).

    A draft may be exported only when the caller asks for it (``allow_draft``), the deployment
    permits it (``settings.allow_draft_export``) and ``settings.pilot`` is false; such an export
    carries the ``watermark_draft`` and ``audit_export`` obligations. Requiring both the caller
    and the deployment to agree means a query parameter on the export route cannot by itself
    produce an unapproved document.
    """
    if _is_approved(minutes):
        return PolicyDecision(
            allowed=True, reason="minutes approved", obligations=["stamp_classification"]
        )
    if minutes.review.status == "approved":
        return PolicyDecision(allowed=False, reason="approved minutes still carry blocking flags")
    if not allow_draft:
        return PolicyDecision(allowed=False, reason="minutes are not approved")
    if settings is None or settings.pilot:
        return PolicyDecision(allowed=False, reason="draft export is disabled under the pilot")
    if not settings.allow_draft_export:
        return PolicyDecision(allowed=False, reason="draft export is disabled in this deployment")
    return PolicyDecision(
        allowed=True,
        reason="draft export explicitly requested outside the pilot",
        obligations=["watermark_draft", "audit_export", "stamp_classification"],
    )


def detokenise_allowed(minutes: Minutes) -> PolicyDecision:
    """Identifiers may be restored only in approved, non-restricted minutes (C-06)."""
    if minutes.classification is Classification.restricted:
        return PolicyDecision(allowed=False, reason="restricted minutes are never de-tokenised")
    if not _is_approved(minutes):
        return PolicyDecision(allowed=False, reason="de-tokenisation requires approved minutes")
    obligations = (
        ["organiser_confirmation"] if minutes.classification is Classification.confidential else []
    )
    return PolicyDecision(allowed=True, reason="approved minutes", obligations=obligations)


def evaluate(action: str, ctx: dict[str, Any], settings: Settings) -> PolicyDecision:
    """Dispatch ``action`` to the matching rule; unknown actions and missing context are refused.

    Actions: ``"classification"`` (ctx ``classification``), ``"export"`` (ctx ``minutes``,
    optional ``allow_draft``), ``"detokenise"`` (ctx ``minutes``).
    """
    try:
        if action == "classification":
            return classification_allowed(ctx["classification"], settings)
        if action == "export":
            return export_allowed(
                ctx["minutes"], settings, allow_draft=bool(ctx.get("allow_draft", False))
            )
        if action == "detokenise":
            return detokenise_allowed(ctx["minutes"])
    except KeyError as exc:
        return PolicyDecision(allowed=False, reason=f"missing context {exc.args[0]!r}")
    return PolicyDecision(allowed=False, reason=f"unknown policy action {action!r}")
