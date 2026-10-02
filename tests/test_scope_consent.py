"""Scope exclusions and the consent gate (controls C-03, C-09, C-10)."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO
from helpers_core import make_audit, make_meeting, make_settings

from praktika import consent, retention, scope
from praktika.config import Settings
from praktika.errors import ClassificationNotAllowed, ConsentRefused, ScopeError
from praktika.identity import FakeIdentity
from praktika.models import Classification, ConsentRecord, MeetingType

ALL_CHECKS = {k: True for k in scope.scope_check_keys()}
PURPOSE = "Minutes of the data team weekly meeting"


def _gate(tmp_path: Path, meeting: Any = None, settings: Settings | None = None, **over: Any):
    audit, path = make_audit(tmp_path)
    kwargs: dict[str, Any] = {
        "notified": True,
        "objections": False,
        "method": "spoken",
        "teams_transcription_started": True,
        "purpose": PURPOSE,
        "scope_checks": dict(ALL_CHECKS),
        "identity": FakeIdentity(source="session").current(),
        "settings": settings or make_settings(tmp_path),
        "audit": audit,
    }
    kwargs.update(over)
    return consent.gate(meeting or make_meeting(), **kwargs), path


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- texts


def test_consent_texts_are_versioned_bilingual_and_organisation_neutral() -> None:
    """Any change to the wording needs a new SCRIPT_VERSION; the texts name no organisation."""
    assert consent.SCRIPT_VERSION == "2026-10-02"
    assert len(consent.BANNER_EN) <= consent.BANNER_MAX_CHARS == 200
    assert len(consent.BANNER_AR) <= consent.BANNER_MAX_CHARS
    assert re.search(r"[؀-ۿ]", consent.SCRIPT_AR) and not re.search(r"[؀-ۿ]", consent.SCRIPT_EN)
    assert re.search(r"[؀-ۿ]", consent.BANNER_AR) and not re.search(r"[؀-ۿ]", consent.BANNER_EN)
    assert "our internal AI notetaker, which runs only on our own systems" in consent.SCRIPT_EN
    assert consent.BANNER_EN.startswith("Our internal AI notetaker is recording")
    assert "أنظمتنا الخاصة" in consent.SCRIPT_AR and "لدينا" in consent.BANNER_AR


def test_consent_texts_promise_no_more_than_the_default_retention_does() -> None:
    """The notice participants hear must be true of the code with the default settings
    (pilot, so Internal only; local mode): audio goes at approval or discard and within the
    Internal audio timer, the transcript within the transcript timer after approval and within
    the hard maximum otherwise, a legal hold can keep both, and only approved minutes leave."""
    default = {name: field.default for name, field in Settings.model_fields.items()}
    assert default["pilot"] and default["mode"] == "local"
    assert default["retention_audio_hours"]["internal"] == 24
    assert default["retention_transcript_days"]["internal"] == 14
    assert retention.TRANSCRIPT_MAX_DAYS["internal"] == 60
    assert default["allow_draft_export"] is False
    script = consent.SCRIPT_EN
    assert "The recording is deleted after the minutes are approved or discarded" in script
    assert "within about a day in any case" in script
    assert "The transcript is deleted within two weeks of the minutes being approved" in script
    assert "and within 60 days in any case" in script
    assert "A legal hold can require us to keep them longer" in script
    assert "I check and approve the minutes before they are shared" in script
    for overclaim in ("after transcription", "only until", "before anyone sees", "On-prem"):
        assert overclaim not in script and overclaim not in consent.BANNER_EN
    assert "within about a day, transcript within 60 days, unless on legal hold" in (
        consent.BANNER_EN
    )
    # the Arabic texts carry the same promises: a day, two weeks, sixty days, a legal hold
    assert "خلال يوم تقريباً" in consent.SCRIPT_AR and "خلال يوم تقريباً" in consent.BANNER_AR
    assert "خلال أسبوعين من اعتماد المحضر" in consent.SCRIPT_AR
    assert "ستين يوماً" in consent.SCRIPT_AR and "ستين يوماً" in consent.BANNER_AR
    assert "حفظ قانوني" in consent.SCRIPT_AR and "حفظ قانوني" in consent.BANNER_AR
    assert "بعد التفريغ" not in consent.SCRIPT_AR + consent.BANNER_AR


def test_controls_md_quotes_the_consent_texts_and_version_verbatim() -> None:
    """docs/CONTROLS.md reproduces the four texts and their version exactly as the code has
    them, so the published notice and the one `praktika consent-script` prints never drift."""
    controls = (REPO / "docs" / "CONTROLS.md").read_text(encoding="utf-8")
    section = controls.split("\n## Consent script and chat notice\n", 1)[1].split("\n## ", 1)[0]
    assert f"(`{consent.SCRIPT_VERSION}`)" in section
    for text in (consent.SCRIPT_EN, consent.BANNER_EN, consent.SCRIPT_AR, consent.BANNER_AR):
        assert f"\n> {text}\n" in section


def test_print_script_and_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    class Console:
        def print(self, text: str = "") -> None:
            printed.append(text)

    consent.print_script(Console(), "ar")
    assert consent.SCRIPT_AR in printed and consent.BANNER_AR in printed
    with pytest.raises(ValueError):
        consent.print_script(Console(), "fr")  # type: ignore[arg-type]

    copied: list[bytes] = []

    def fake_run(args: list[str], **kw: Any) -> Any:
        copied.append(kw["input"])
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(consent.subprocess, "run", fake_run)
    monkeypatch.setattr(consent, "PBCOPY", __file__)  # a clipboard tool "exists" on any host
    assert consent.copy_banner_to_clipboard("en") is True
    assert copied == [consent.BANNER_EN.encode("utf-8")]
    assert consent.copy_banner_to_clipboard("both") is True
    assert copied[-1] == (consent.BANNER_EN + "\n" + consent.BANNER_AR).encode("utf-8")

    def broken(*a: Any, **kw: Any) -> Any:
        raise FileNotFoundError("pbcopy")

    monkeypatch.setattr(consent.subprocess, "run", broken)
    assert consent.copy_banner_to_clipboard("en") is False
    assert consent.copy_banner_to_clipboard("xx") is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------- scope


def test_scope_checklist_is_bilingual_and_complete() -> None:
    items = scope.scope_checklist()
    assert {k for k, _, _ in items} == scope.scope_check_keys()
    assert {"not_board", "not_foreign_hosted", "not_hr", "not_customer_call"} <= set(ALL_CHECKS)
    for _, en, ar in items:
        assert en and re.search(r"[؀-ۿ]", ar)


@pytest.mark.parametrize("exclusion", sorted(scope.PILOT_EXCLUSIONS))
def test_scope_check_refuses_each_exclusion_as_tag(exclusion: str, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(ScopeError) as info:
        scope.check("general", {"weekly", exclusion.upper()}, settings)
    assert re.search(r"[؀-ۿ]", str(info.value)), "reason must be bilingual"
    assert exclusion.replace("_", " ") in str(info.value)
    scope.check("general", {"weekly"}, settings)  # no exclusion: no error


def test_scope_check_refuses_board_meeting_types_and_respects_pilot_flag(tmp_path: Path) -> None:
    with pytest.raises(ScopeError):
        scope.check("board_subcommittee", set(), make_settings(tmp_path))
    with pytest.raises(ScopeError):
        scope.check("Board", set(), make_settings(tmp_path))
    scope.check("board", {"hr"}, make_settings(tmp_path, pilot=False))


# --------------------------------------------------------------------------- gate


def test_gate_refuses_without_notified(tmp_path: Path) -> None:
    with pytest.raises(ConsentRefused):
        _gate(tmp_path, notified=False)
    events = _events(tmp_path / "audit.jsonl")
    assert [e["event"] for e in events] == ["scope.refused"]
    assert events[0]["detail"]["reason"] == "recording not used"


def test_gate_refuses_with_objections(tmp_path: Path) -> None:
    with pytest.raises(ConsentRefused, match="objection"):
        _gate(tmp_path, objections=True)
    text = (tmp_path / "audit.jsonl").read_text("utf-8")
    assert "recording not used" in text
    assert "objector" not in text and "who" not in json.loads(text)["detail"]
    assert "objection" not in text, "the audit trail never says that someone objected"
    assert json.loads(text)["detail"] == {"reason": "recording not used"}
    # every refusal carries the same neutral payload, so refusals are indistinguishable
    with pytest.raises(ConsentRefused):
        _gate(tmp_path / "b", notified=False)
    other = (tmp_path / "b" / "audit.jsonl").read_text("utf-8")
    assert json.loads(other)["detail"] == json.loads(text)["detail"]


@pytest.mark.parametrize("exclusion", sorted(scope.PILOT_EXCLUSIONS))
def test_gate_refuses_each_pilot_exclusion(exclusion: str, tmp_path: Path) -> None:
    with pytest.raises(ScopeError):
        _gate(tmp_path, make_meeting(tags={exclusion}))
    events = _events(tmp_path / "audit.jsonl")
    assert events[-1]["event"] == "scope.refused"
    assert events[-1]["detail"] == {"reason": "recording not used"}, "no check name audited"


def test_gate_refuses_confidential_in_pilot(tmp_path: Path) -> None:
    with pytest.raises(ClassificationNotAllowed, match="pilot"):
        _gate(tmp_path, make_meeting(classification=Classification.confidential))
    assert _events(tmp_path / "audit.jsonl")[-1]["classification"] == "confidential"


def test_gate_refuses_restricted_in_local_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, pilot=False, mode="local")
    with pytest.raises(ClassificationNotAllowed, match="service"):
        _gate(tmp_path, make_meeting(classification=Classification.restricted), settings)


def test_gate_refuses_incomplete_or_declined_checklist(tmp_path: Path) -> None:
    partial = dict(ALL_CHECKS)
    partial.pop("not_hr")
    with pytest.raises(ConsentRefused, match="not_hr"):
        _gate(tmp_path, scope_checks=partial)
    declined = {**ALL_CHECKS, "not_board": False}
    with pytest.raises(ConsentRefused, match="not_board"):
        _gate(tmp_path, scope_checks=declined)


def test_gate_refuses_bad_method_or_short_purpose(tmp_path: Path) -> None:
    with pytest.raises(ConsentRefused, match="invalid"):
        _gate(tmp_path, method="telepathy")
    with pytest.raises(ConsentRefused, match="invalid"):
        _gate(tmp_path, purpose="short")


def test_gate_records_and_audits(tmp_path: Path, frozen_clock: Any) -> None:
    record, path = _gate(tmp_path)
    assert isinstance(record, ConsentRecord)
    assert record.meeting_id == "M-20260916-a1b2"
    assert record.script_version == consent.SCRIPT_VERSION
    assert record.recorded_by == "f.khalid@acme.test"
    assert record.recorded_by_source == "session"
    assert record.recorded_at.isoformat().startswith("2026-09-16T06:00:00")
    assert record.scope_checks == ALL_CHECKS and record.purpose == PURPOSE
    events = _events(path)
    assert [e["event"] for e in events] == ["consent.recorded"]
    e = events[0]
    assert e["meeting_id"] == record.meeting_id and e["classification"] == "internal"
    assert e["actor"] == "f.khalid@acme.test" and e["actor_source"] == "session"
    assert e["detail"]["method"] == "spoken" and e["detail"]["teams_transcription_started"]
    assert e["prev_hash"] == "0" * 64 and len(e["hash"]) == 64


def test_gate_fake_identity_recorded_as_local(tmp_path: Path) -> None:
    record, _ = _gate(tmp_path, identity=FakeIdentity().current())
    assert record.recorded_by_source == "local"


def test_gate_accepts_one_to_one_and_mancom_types(tmp_path: Path) -> None:
    for mt in (MeetingType.one_to_one, MeetingType.mancom):
        record, _ = _gate(tmp_path, make_meeting(meeting_type=mt))
        assert record.notified


def test_no_skip_flag_exists() -> None:
    suspicious = re.compile(r"skip|bypass|force|no_consent|noconsent|unsafe|override", re.I)
    params = inspect.signature(consent.gate).parameters
    assert not [p for p in params if suspicious.search(p)], "gate must have no skip parameter"
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in params.items() if name != "meeting"
    )
    assert not [f for f in Settings.model_fields if suspicious.search(f)]
    source = inspect.getsource(consent)
    assert "os.environ" not in source and "getenv" not in source, "gate must not read env"
    assert not re.search(r"def gate\([^)]*=\s*(True|False)", source, re.S), "no defaults"
