"""The evaluation gate on an English-only golden set (deployment: English meetings).

Regression: the Arabic/English decision-recall gap was a hard criterion and ``None`` failed it,
so a golden set with no Arabic meeting could never pass the gate. The criterion now applies only
when the set holds a non-English meeting and is reported as "not applicable" otherwise.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praktika.cli import app
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.eval import checks, golden
from praktika.eval import report as report_mod

PERFECT: dict[str, float] = {m: 1.0 for m in checks.METRICS} | {"unsupported_claim_rate": 0.0}


def good(name: str, language: str = "en", **over: Any) -> dict[str, Any]:
    """A synthetic per-meeting score that meets every threshold."""
    return {
        "name": name,
        "language": language,
        "schema_ok": True,
        **PERFECT,
        "uncited_decisions_in_body": 0,
        **over,
    }


def english_report() -> checks.EvalReport:
    return checks.aggregate([good("01_en"), good("02_en"), good("03_en")])


def test_english_only_gate_passes_when_every_other_criterion_does() -> None:
    report = english_report()
    assert report.ar_en_recall_gap is None
    assert not checks.language_gap_applicable(report)
    passed, reasons = checks.gate(report)
    assert passed and reasons == []
    # The underlying report gate still counts the unmeasurable gap as a failure: that is the
    # behaviour this wrapper exists to correct for English-only sets.
    assert report_mod.gate(report)[1] == ["ar_en_recall_gap = None > 10.0"]


def test_english_only_gate_still_fails_on_other_criteria() -> None:
    report = checks.aggregate([good("01_en"), good("02_en", decision_precision=0.5)])
    passed, reasons = checks.gate(report)
    assert not passed
    assert reasons == ["decision_precision = 0.75 < 0.95"]
    assert not any(r.startswith("ar_en_recall_gap") for r in reasons)


def test_english_only_threshold_override_still_applies() -> None:
    passed, reasons = checks.gate(english_report(), {"decision_recall": 1.01})
    assert not passed and reasons == ["decision_recall = 1.0 < 1.01"]


def test_empty_evaluation_still_fails() -> None:
    passed, reasons = checks.gate(checks.aggregate([]))
    assert not passed and any("decision_precision" in r for r in reasons)
    assert not any(r.startswith("ar_en_recall_gap") for r in reasons)


def test_gap_criterion_applies_once_an_arabic_meeting_is_present() -> None:
    wide = checks.aggregate([good("01_en"), good("02_ar", "ar-mixed", decision_recall=0.95)])
    assert checks.language_gap_applicable(wide)
    assert wide.ar_en_recall_gap == pytest.approx(5.0)
    passed, reasons = checks.gate(wide, {"ar_en_recall_gap_max": 3.0})
    assert not passed and len(reasons) == 1 and reasons[0].startswith("ar_en_recall_gap = 5.0")
    assert checks.gate(wide)[0], "a 5-point gap is inside the default 10-point limit"


def test_unmeasurable_gap_with_an_arabic_meeting_still_fails() -> None:
    """An Arabic meeting with nothing scoreable leaves the gap ``None``: that stays a failure."""
    report = checks.aggregate([good("01_en"), good("02_ar", "ar-mixed", decision_recall=None)])
    assert report.ar_en_recall_gap is None
    passed, reasons = checks.gate(report)
    assert not passed and reasons == ["ar_en_recall_gap = None > 10.0"]


def test_english_only_report_says_not_applicable(tmp_path: Path) -> None:
    out = checks.write_report(english_report(), tmp_path / "eval_report.md")
    text = out.read_text(encoding="utf-8")
    assert "Gate: PASS" in text and "Gate: FAIL" not in text
    assert "ar_en_recall_gap (points) | not applicable" in text
    assert "Skipped: ar_en_recall_gap is not applicable" in text
    assert "ar_en_recall_gap = None" not in text
    assert all(f"| {n} | en |" in text for n in ("01_en", "02_en", "03_en"))


def test_english_only_failing_report_lists_reasons_and_the_skip(tmp_path: Path) -> None:
    report = checks.aggregate([good("01_en", action_f1=0.5)])
    text = checks.write_report(report, tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Gate: FAIL" in text and "- action_f1 = 0.5 < 0.85" in text
    assert "ar_en_recall_gap = None" not in text and "not applicable" in text


def test_mixed_report_keeps_the_numeric_gap(tmp_path: Path) -> None:
    mixed = checks.aggregate([good("01_en"), good("02_ar", "ar-mixed")])
    text = checks.write_report(mixed, tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Gate: PASS" in text and "| ar_en_recall_gap (points) | 0.000 |" in text
    assert "not applicable" not in text


@pytest.fixture
def english_golden(tmp_path: Path) -> Path:
    """The English meetings of the shipped golden set, copied into their own directory."""
    target = tmp_path / "golden_en"
    target.mkdir()
    for gm in golden.load_all():
        if gm.language == "en":
            shutil.copytree(golden.GOLDEN_DIR / gm.name, target / gm.name)
    assert len(list(target.iterdir())) == 4
    return target


def test_eval_cli_passes_on_english_only_golden_set(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, english_golden: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(ctx, "load_settings", lambda: tmp_settings)
    out = tmp_path / "eval_report.md"
    result = CliRunner().invoke(
        app, ["eval", "--llm", "fake", "--golden", str(english_golden), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Evaluated 4 meeting(s)" in result.output and "Gate: PASS" in result.output
    text = out.read_text(encoding="utf-8")
    assert "Gate: PASS" in text and "ar_en_recall_gap (points) | not applicable" in text
