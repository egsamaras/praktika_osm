"""Policy rules (controls C-06, C-07, C-09, C-15)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
from helpers_core import NOW, make_settings
from helpers_foundation import minutes

from praktika import policy
from praktika.models import Classification, Flag, Minutes, Review

SRC = Path(__file__).resolve().parent.parent / "src" / "praktika"


def _approved(**over: object) -> Minutes:
    review = Review(
        status="approved",
        reviewer="F. Khalid",
        reviewer_source="session",
        reviewed_at=datetime.fromisoformat(NOW.isoformat()),
    )
    return minutes(review=review, **over)


@pytest.mark.parametrize(
    ("pilot", "mode", "cls", "allowed"),
    [
        (True, "local", "internal", True),
        (True, "local", "confidential", False),
        (True, "local", "restricted", False),
        (True, "service", "confidential", False),
        (False, "local", "internal", True),
        (False, "local", "confidential", False),
        (False, "local", "restricted", False),
        (False, "service", "internal", True),
        (False, "service", "confidential", True),
        (False, "service", "restricted", True),
    ],
)
def test_classification_rules(
    tmp_path: Path, pilot: bool, mode: str, cls: str, allowed: bool
) -> None:
    settings = make_settings(tmp_path, pilot=pilot, mode=mode)
    decision = policy.classification_allowed(Classification(cls), settings)
    assert decision.allowed is allowed, decision.reason
    assert policy.classification_allowed(cls, settings).allowed is allowed
    if allowed and cls == "restricted":
        assert "no_fts_index" in decision.obligations
    if not allowed:
        assert decision.reason


def test_unknown_classification_denied(tmp_path: Path) -> None:
    assert not policy.classification_allowed("public", make_settings(tmp_path)).allowed


def test_export_requires_approval(tmp_path: Path) -> None:
    pilot = make_settings(tmp_path)
    assert not policy.export_allowed(minutes(), pilot).allowed
    assert not policy.export_allowed(minutes(), pilot, allow_draft=True).allowed
    assert not policy.export_allowed(minutes(), None, allow_draft=True).allowed
    in_review = minutes(review=Review(status="in_review"))
    assert not policy.export_allowed(in_review, pilot).allowed
    ok = policy.export_allowed(_approved(), pilot)
    assert ok.allowed and "stamp_classification" in ok.obligations
    # Approved on paper but with an open priority-1 flag: still refused.
    blocked = _approved(flags=[Flag(kind="uncited_item_removed", detail="x", priority=1)])
    assert not policy.export_allowed(blocked, pilot).allowed
    cleared = _approved(
        flags=[Flag(kind="uncited_item_removed", detail="x", priority=1, cleared_by="F. Khalid")]
    )
    assert policy.export_allowed(cleared, pilot).allowed
    # Draft export outside the pilot, explicitly requested: allowed with obligations.
    prod = make_settings(tmp_path, pilot=False, mode="service")
    assert not policy.export_allowed(minutes(), prod, allow_draft=True).allowed  # no opt-in
    prod = prod.model_copy(update={"allow_draft_export": True})
    draft = policy.export_allowed(minutes(), prod, allow_draft=True)
    assert draft.allowed and {"watermark_draft", "audit_export"} <= set(draft.obligations)
    assert not policy.export_allowed(minutes(), prod).allowed


def test_detokenise_denied_for_restricted() -> None:
    assert not policy.detokenise_allowed(minutes()).allowed
    assert not policy.detokenise_allowed(
        _approved(classification=Classification.restricted)
    ).allowed
    internal = policy.detokenise_allowed(_approved())
    assert internal.allowed and internal.obligations == []
    confidential = policy.detokenise_allowed(_approved(classification=Classification.confidential))
    assert confidential.allowed and confidential.obligations == ["organiser_confirmation"]


def test_evaluate_dispatch_and_default_deny(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert policy.evaluate("classification", {"classification": "internal"}, settings).allowed
    assert not policy.evaluate("classification", {"classification": "restricted"}, settings).allowed
    assert not policy.evaluate("export", {"minutes": minutes()}, settings).allowed
    assert policy.evaluate("export", {"minutes": _approved()}, settings).allowed
    assert policy.evaluate("detokenise", {"minutes": _approved()}, settings).allowed
    assert not policy.evaluate("send_email", {"minutes": _approved()}, settings).allowed
    missing = policy.evaluate("export", {}, settings)
    assert not missing.allowed and "minutes" in missing.reason
    assert isinstance(missing, policy.PolicyDecision)


def test_no_analytics_fields() -> None:
    forbidden = re.compile(r"\b(talk_time|sentiment|attendance_score)\b")
    offenders = [
        f"{p.relative_to(SRC)}:{i}"
        for p in sorted(SRC.rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if forbidden.search(line)
    ]
    assert offenders == [], f"per-person analytics are prohibited (C-15): {offenders}"
