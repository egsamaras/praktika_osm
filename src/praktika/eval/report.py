"""Evaluation report, rollout gates and ``eval_report.md``.

``aggregate`` folds per-meeting scores (``eval.checks.score``) into an ``EvalReport`` (macro
averages, the AR/EN decision-recall gap in points, schema pass rate); ``gate`` applies the
rollout thresholds and reports every breach; ``write_report`` renders the Markdown report.
A metric that is ``None`` because nothing was scoreable fails its gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

DEFAULT_THRESHOLDS: dict[str, float] = {
    "decision_precision": 0.95,
    "decision_recall": 0.90,
    "action_f1": 0.85,
    "owner_accuracy": 0.90,
    "trap_resistance": 0.95,
    "citation_validity": 0.98,
    "numbers_present_or_flagged": 1.0,
    "names_resolved_or_flagged": 1.0,
    "unsupported_claim_rate_max": 0.02,
    "unsupported_decisions_max": 0,
    "ar_en_recall_gap_max": 10.0,
    "schema_pass_rate": 1.0,
}
METRICS = (
    "decision_precision",
    "decision_recall",
    "action_precision",
    "action_recall",
    "action_f1",
    "owner_accuracy",
    "due_accuracy",
    "trap_resistance",
    "citation_validity",
    "numbers_present_or_flagged",
    "names_resolved_or_flagged",
    "unsupported_claim_rate",
)


class MeetingScore(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    language: str
    schema_ok: bool = True
    uncited_decisions_in_body: int = 0


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meetings: list[MeetingScore]
    aggregate: dict[str, float | None]
    ar_en_recall_gap: float | None
    schema_pass_rate: float
    unsupported_decisions: int


def failed_score(name: str, language: str) -> dict[str, Any]:
    """The score of a meeting whose output failed schema validation (counts against every gate)."""
    zeros = {m: 0.0 for m in METRICS}
    zeros["unsupported_claim_rate"] = 1.0
    return {
        "name": name,
        "language": language,
        "schema_ok": False,
        **zeros,
        "uncited_decisions_in_body": 0,
    }


def _mean(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def aggregate(scores: list[dict[str, Any]]) -> EvalReport:
    """Macro-average the metrics; the AR/EN gap is |recall(en) - recall(ar-mixed)| in points."""
    agg = {m: _mean([s.get(m) for s in scores]) for m in METRICS}
    en = _mean([s.get("decision_recall") for s in scores if s["language"] == "en"])
    ar = _mean([s.get("decision_recall") for s in scores if s["language"] != "en"])
    gap = abs(en - ar) * 100 if en is not None and ar is not None else None
    return EvalReport(
        meetings=[MeetingScore(**s) for s in scores],
        aggregate=agg,
        ar_en_recall_gap=gap,
        schema_pass_rate=(sum(1 for s in scores if s["schema_ok"]) / len(scores))
        if scores
        else 0.0,
        unsupported_decisions=sum(int(s.get("uncited_decisions_in_body", 0)) for s in scores),
    )


def gate(report: EvalReport, thresholds: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """Apply the rollout thresholds; returns (passed, reasons). A metric that is ``None`` because
    nothing was scoreable fails the gate: an empty evaluation is not a passing one."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []
    for key, minimum in t.items():
        if key.endswith("_max"):
            continue
        if key == "schema_pass_rate":
            value: float | None = report.schema_pass_rate
        else:
            value = report.aggregate.get(key)
        if value is None or value < minimum:
            reasons.append(f"{key} = {value} < {minimum}")
    rate = report.aggregate.get("unsupported_claim_rate")
    if rate is None or rate > t["unsupported_claim_rate_max"]:
        reasons.append(f"unsupported_claim_rate = {rate} > {t['unsupported_claim_rate_max']}")
    if report.unsupported_decisions > t["unsupported_decisions_max"]:
        reasons.append(f"unsupported_decisions = {report.unsupported_decisions}")
    if report.ar_en_recall_gap is None or report.ar_en_recall_gap > t["ar_en_recall_gap_max"]:
        reasons.append(
            f"ar_en_recall_gap = {report.ar_en_recall_gap} > {t['ar_en_recall_gap_max']}"
        )
    return not reasons, reasons


def write_report(report: EvalReport, path: Path) -> Path:
    """Write ``eval_report.md`` (aggregate table, gate result, one row per meeting)."""
    passed, reasons = gate(report)
    lines = ["# Praktika evaluation report", "", f"Gate: {'PASS' if passed else 'FAIL'}", ""]
    lines += [f"- {r}" for r in reasons]
    lines += ["", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in report.aggregate.items()]
    lines += [
        f"| ar_en_recall_gap (points) | {_fmt(report.ar_en_recall_gap)} |",
        f"| schema_pass_rate | {_fmt(report.schema_pass_rate)} |",
        f"| unsupported_decisions | {report.unsupported_decisions} |",
        "",
        "| Meeting | Lang | Dec P | Dec R | Act F1 | Traps | Cites | Numbers | Names |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in report.meetings:
        d = m.model_dump()
        lines.append(
            f"| {m.name} | {m.language} | {_fmt(d.get('decision_precision'))} | "
            f"{_fmt(d.get('decision_recall'))} | {_fmt(d.get('action_f1'))} | "
            f"{_fmt(d.get('trap_resistance'))} | {_fmt(d.get('citation_validity'))} | "
            f"{_fmt(d.get('numbers_present_or_flagged'))} | "
            f"{_fmt(d.get('names_resolved_or_flagged'))} |"
        )
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"
