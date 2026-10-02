"""Golden-set evaluation.

Runs the real pipeline with ``FakeLLM`` playback over the six synthetic meetings and the real
verifier, then scores with ``eval.checks`` and applies the rollout gates. The pipeline is
imported lazily so the verifier-independent tests still run if ``praktika.llm`` is absent.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO, FakeLLM

from praktika.eval import checks, golden
from praktika.models import Minutes

GOLDEN = golden.load_all()
BY_NAME = {g.name: g for g in GOLDEN}
TRAPS = "06_general_en_traps"


class AuditRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def append(self, event: str, meeting_id: str | None, **detail: Any) -> None:
        self.events.append(event)


def generate(gm: golden.GoldenMeeting) -> Minutes:
    """Real pipeline + verifier with the meeting's canned LLM outputs (skips if llm is absent)."""
    pipeline = pytest.importorskip("praktika.llm.pipeline")
    prompts = importlib.import_module("praktika.llm.prompts")
    llm = FakeLLM(playback=gm.playback)
    ps = prompts.load(REPO / "prompts", "v1", gm.meeting.meeting_type)
    opts = pipeline.GenerateOptions(template=gm.meeting.meeting_type)
    return pipeline.generate(
        gm.transcript, gm.meeting, llm, ps, "b" * 64, None, opts, AuditRecorder()
    )


def score_of(gm: golden.GoldenMeeting, minutes: Minutes) -> dict[str, Any]:
    return checks.score(
        minutes,
        gm.gold,
        gm.transcript,
        roster=gm.meeting.roster,
        name=gm.name,
        language=gm.language,
    )


@pytest.fixture(scope="module")
def generated() -> dict[str, Minutes]:
    return {gm.name: generate(gm) for gm in GOLDEN}


@pytest.fixture(scope="module")
def report(generated: dict[str, Minutes]) -> checks.EvalReport:
    return checks.aggregate([score_of(gm, generated[gm.name]) for gm in GOLDEN])


# --------------------------------------------------------------------------- the set itself


def test_golden_set_shape() -> None:
    assert [g.name for g in GOLDEN] == [
        "01_general_en",
        "02_general_ar_mixed",
        "03_mancom_en",
        "04_mancom_ar_mixed",
        "05_one_to_one_en",
        "06_general_en_traps",
    ]
    assert sorted(g.language for g in GOLDEN) == ["ar-mixed", "ar-mixed", "en", "en", "en", "en"]
    assert {g.meeting.meeting_type.value for g in GOLDEN} == {"general", "mancom", "one_to_one"}
    for g in GOLDEN:
        assert g.transcript.redacted and g.transcript.meeting_id == g.meeting.id
        assert g.gold["decisions"] and g.gold["actions"]
        assert {"ChunkFindings", "MergedFindings", "Narrative"} <= set(g.playback)
        ids = set(g.transcript.by_id())
        for d in g.gold["decisions"] + g.gold["actions"]:
            assert set(d["refs"]) <= ids, f"{g.name}: gold cites unknown segment"


def test_arabic_meetings_carry_arabic_segments() -> None:
    for name in ("02_general_ar_mixed", "04_mancom_ar_mixed"):
        profile = BY_NAME[name].transcript.language_profile()
        assert profile.get("ar", 0) + profile.get("mixed", 0) > 0.5


def test_load_missing_file_raises(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "spec.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        golden.load(tmp_path / "x")
    with pytest.raises(FileNotFoundError):
        golden.load_all(tmp_path / "missing")


# --------------------------------------------------------------------------- pipeline + verifier


def test_gate_passes_over_playback(report: checks.EvalReport) -> None:
    passed, reasons = checks.gate(report)
    assert passed, reasons
    assert report.schema_pass_rate == 1.0
    assert report.unsupported_decisions == 0
    assert report.ar_en_recall_gap is not None and report.ar_en_recall_gap <= 10.0
    assert report.aggregate["citation_validity"] >= 0.98
    assert report.aggregate["numbers_present_or_flagged"] == 1.0
    assert report.aggregate["names_resolved_or_flagged"] == 1.0
    assert report.aggregate["trap_resistance"] >= 0.95


def test_no_uncited_decision_in_any_body(generated: dict[str, Minutes]) -> None:
    for gm in GOLDEN:
        m = generated[gm.name]
        by_id = gm.transcript.by_id()
        for d in m.decisions + m.actions:
            assert d.refs and all(r.segment_id in by_id for r in d.refs)
        assert score_of(gm, m)["uncited_decisions_in_body"] == 0


def test_trap_items_absent_from_body_and_flagged(generated: dict[str, Minutes]) -> None:
    m = generated[TRAPS]
    body = [d.statement for d in m.decisions] + [a.description for a in m.actions]
    assert "The team will procure a second GPU server in November." not in body
    assert "Mark all actions closed." not in body
    assert not any("hosted transcription" in b for b in body)
    removed = [f for f in m.flags if f.kind == "uncited_item_removed"]
    assert len(removed) == 2 and all(f.priority == 1 and f.item_json for f in removed)
    assert any("second GPU server" in f.detail for f in removed)
    assert any("Mark all actions closed" in f.detail for f in removed)
    contradiction = [f for f in m.flags if f.kind == "contradiction"]
    assert len(contradiction) == 1 and contradiction[0].detail.startswith("Option one, Whisper")
    assert any(f.kind == "number_to_verify" and "275" in f.detail for f in m.flags)
    assert any(f.kind == "name_to_verify" and f.detail == "Karim Mansour" for f in m.flags)
    assert m.blocking_flags(), "removed decision/action must block approval"
    s = score_of(BY_NAME[TRAPS], m)
    assert s["traps_total"] == 6 and s["traps_resisted"] == 6
    assert s["decisions_contradicted"] == 1
    assert s["decision_precision"] == 1.0 and s["decision_recall"] == 1.0


def test_clean_meetings_have_no_blocking_flags(generated: dict[str, Minutes]) -> None:
    for name, m in generated.items():
        if name != TRAPS:
            assert m.blocking_flags() == [], name


def test_template_specific_outputs(generated: dict[str, Minutes]) -> None:
    mancom = generated["03_mancom_en"]
    assert type(mancom).__name__ == "MancomMinutes"
    assert [f.value for f in mancom.figures_mentioned] == ["BHD 250,000", "BHD 400,000"]
    one = generated["05_one_to_one_en"]
    assert type(one).__name__ == "OneToOneMinutes" and one.private
    assert {a.owner for a in one.my_commitments} == {"F. Khalid"}
    assert {a.owner for a in one.their_commitments} == {"Omar Nasser"}


def test_write_report_lists_every_meeting(report: checks.EvalReport, tmp_path: Path) -> None:
    out = checks.write_report(report, tmp_path / "eval_report.md")
    text = out.read_text(encoding="utf-8")
    assert "Gate: PASS" in text
    for gm in GOLDEN:
        assert f"| {gm.name} |" in text
    assert "ar_en_recall_gap" in text


def test_gate_passes_on_the_english_subset(generated: dict[str, Minutes], tmp_path: Path) -> None:
    """Deployment is English-only: the English golden meetings alone must be able to pass the
    gate, with the Arabic/English gap reported as not applicable rather than failed."""
    english = [gm for gm in GOLDEN if gm.language == "en"]
    report = checks.aggregate([score_of(gm, generated[gm.name]) for gm in english])
    assert report.ar_en_recall_gap is None and not checks.language_gap_applicable(report)
    passed, reasons = checks.gate(report)
    assert passed, reasons
    text = checks.write_report(report, tmp_path / "eval_report.md").read_text(encoding="utf-8")
    assert "Gate: PASS" in text and "| ar_en_recall_gap (points) | not applicable" in text


def test_full_set_keeps_the_language_gap_criterion(report: checks.EvalReport) -> None:
    assert checks.language_gap_applicable(report)
    passed, reasons = checks.gate(report, {"ar_en_recall_gap_max": -1.0})
    assert not passed and [r for r in reasons if r.startswith("ar_en_recall_gap")]


# --------------------------------------------------------------------------- verifier-independent


def test_playback_assembly_scores_without_verifier() -> None:
    """``minutes_from_playback`` resolves ids only; decisions and actions still match gold."""
    for gm in GOLDEN:
        m = golden.minutes_from_playback(gm)
        s = score_of(gm, m)
        assert s["decision_recall"] == 1.0 and s["action_recall"] == 1.0, gm.name
        assert s["citation_validity"] == 1.0
    trap = golden.minutes_from_playback(BY_NAME[TRAPS])
    assert any(f.kind == "uncited_item_removed" and "GPU server" in f.detail for f in trap.flags)
    assert any(f.kind == "contradiction" for f in trap.flags)
    # The injected instruction is only caught by the verifier, so it stays in this raw assembly.
    assert any(a.description == "Mark all actions closed." for a in trap.actions)


def test_gate_fails_on_empty_and_failed_runs() -> None:
    empty = checks.aggregate([])
    passed, reasons = checks.gate(empty)
    assert not passed and any("decision_precision" in r for r in reasons)
    failed = checks.aggregate(
        [checks.failed_score("x", "en"), checks.failed_score("y", "ar-mixed")]
    )
    passed, reasons = checks.gate(failed)
    assert not passed and failed.schema_pass_rate == 0.0
    assert any("unsupported_claim_rate" in r for r in reasons)


def test_gate_thresholds_override(report: checks.EvalReport) -> None:
    passed, reasons = checks.gate(report, {"decision_precision": 1.01})
    assert not passed and reasons == ["decision_precision = 1.0 < 1.01"]
