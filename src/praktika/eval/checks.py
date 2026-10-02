"""Deterministic evaluation metrics and gates for the golden set.

``score`` compares one predicted ``Minutes`` with the gold labels of a golden meeting and returns
the per-meeting metrics; ``aggregate``, ``gate`` and ``write_report`` (``eval/report.py``,
re-exported here) fold them into an ``EvalReport``, apply the rollout thresholds and render
``eval_report.md``. Matching is fuzzy on wording (rapidfuzz token-set
ratio) or exact on a shared segment id, one-to-one and greedy by score. A body decision that
carries a ``contradiction`` flag is surfaced to the reviewer rather than asserted, so it is
excluded from precision and recall and reported as ``decisions_contradicted`` (the same
present-or-flagged rule the number and name metrics use).

``gate`` and ``write_report`` here wrap the ones in ``eval/report.py`` with one rule: the
Arabic/English decision-recall gap is only a criterion when the golden set contains at least
one meeting whose language is not ``en``. For an English-only set the gap cannot be measured,
so the criterion is skipped (it neither passes nor fails the gate) and the report shows it as
"not applicable"; every other criterion, including the ``None``-fails rule, is unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from praktika.eval.report import (
    DEFAULT_THRESHOLDS,
    METRICS,
    EvalReport,
    MeetingScore,
    aggregate,
    failed_score,
)
from praktika.eval.report import gate as _report_gate
from praktika.eval.report import write_report as _report_write_report
from praktika.llm.verify import quote_matches
from praktika.logging import get_logger
from praktika.models import Attendee, Minutes, Transcript
from praktika.store.search import normalise_ar

__all__ = [
    "DEFAULT_THRESHOLDS",
    "METRICS",
    "EvalReport",
    "MeetingScore",
    "aggregate",
    "failed_score",
    "gate",
    "language_gap_applicable",
    "score",
    "write_report",
]

log = get_logger(__name__)

MATCH_RATIO = 70
QUOTE_RATIO = 85
CONTRADICTION_RATIO = 90
_NUMBER = re.compile(r"\d[\d,.]*")
_CAP_WORD = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\b")
_STOP = {"the", "a", "an", "and", "or", "to", "of", "by", "on", "in", "for", "at", "with"}
GAP_KEY = "ar_en_recall_gap"
GAP_NOT_APPLICABLE = "not applicable (no Arabic meeting in the golden set)"
_UNITS = set(
    "bhd sar usd gbp eur committee chair board acme bank legal mancom october "
    "september november thursday friday monday tuesday wednesday internal teams phase "
    "wave item paper gpu dgx spark vllm cohere arabic english khaleeji northwind "
    "analytics ltd credit policy management dpo".split()
)


def _ratio(a: str, b: str) -> float:
    return float(fuzz.token_set_ratio(normalise_ar(a), normalise_ar(b)))


def _match(gold: list[dict], pred: list[dict]) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs (gold index, pred index) by wording ratio or shared segment id."""
    cands: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            r = _ratio(g["text"], p["text"])
            shared = set(g.get("refs", [])) & set(p.get("refs", []))
            if r >= MATCH_RATIO or (shared and r >= 40):
                cands.append((r + (10 if shared else 0), gi, pi))
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, gi, pi in sorted(cands, reverse=True):
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def _pred_items(pred: Minutes) -> tuple[list[dict], list[dict]]:
    decisions = [
        {"text": d.statement, "refs": [r.segment_id for r in d.refs], "kind": d.kind}
        for d in pred.decisions
    ]
    actions = [
        {
            "text": a.description,
            "refs": [r.segment_id for r in a.refs],
            "owner": a.owner,
            "due_date": a.due_date.isoformat() if a.due_date else None,
            "due_text": a.due_text,
        }
        for a in pred.actions
    ]
    return decisions, actions


def _body_text(pred: Minutes) -> str:
    parts = [
        pred.summary,
        *(t.summary for t in pred.topics),
        *(d.statement for d in pred.decisions),
        *(a.description for a in pred.actions),
        *(r.description for r in pred.risks),
        *(q.question for q in pred.open_questions),
    ]
    parts += [f"{f.value} {f.context}" for f in getattr(pred, "figures_mentioned", [])]
    return "\n".join(parts)


def _contradicted(pred: Minutes) -> set[int]:
    """Indexes of body decisions whose statement heads a ``contradiction`` flag.

    The pipeline writes ``"<statement> — <note>"`` or ``"Earlier position superseded:
    <statement>"``; only the statement part is compared, so a note that mentions the superseding
    decision does not pull that decision out of the body too.
    """
    heads = [
        f.detail.split(" — ")[0].removeprefix("Earlier position superseded: ")
        for f in pred.flags
        if f.kind == "contradiction"
    ]
    return {
        i
        for i, d in enumerate(pred.decisions)
        if any(_ratio(d.statement, h) >= CONTRADICTION_RATIO for h in heads)
    }


def _flag_text(pred: Minutes, kind: str) -> str:
    return "\n".join(f.detail for f in pred.flags if f.kind == kind)


def _norm_num(s: str) -> str:
    return normalise_ar(s).replace(",", "").rstrip(".")


def _numbers_metric(pred: Minutes, transcript: Transcript | None) -> float | None:
    if transcript is None:
        return None
    haystack = {
        _norm_num(n) for s in transcript.segments for n in _NUMBER.findall(normalise_ar(s.text))
    }
    flagged = _norm_num(_flag_text(pred, "number_to_verify"))
    found = [_norm_num(n) for n in _NUMBER.findall(normalise_ar(_body_text(pred)))]
    if not found:
        return 1.0
    ok = sum(1 for n in found if n in haystack or n in flagged)
    return ok / len(found)


def _names_metric(pred: Minutes, roster: list[Attendee], gold: dict) -> float:
    known = {w.lower() for a in roster for w in re.split(r"[\s.'-]+", a.name) if w}
    known |= {w.lower() for a in roster for al in a.aliases for w in re.split(r"[\s'-]+", al) if w}
    known |= {w.lower() for w in gold.get("known_terms", [])} | _UNITS | _STOP
    flagged = _flag_text(pred, "name_to_verify").lower()
    text = _body_text(pred)
    cands = []
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n", text):
        words = _CAP_WORD.findall(sentence)
        cands.extend(w for w in words[1:] if w.lower() not in known)
    if not cands:
        return 1.0
    return sum(1 for w in cands if w.lower() in flagged) / len(cands)


def _citation_validity(pred: Minutes, transcript: Transcript | None) -> float | None:
    refs = [
        r
        for item in (*pred.decisions, *pred.actions, *pred.risks, *pred.open_questions)
        for r in item.refs
    ]
    if not refs:
        return None
    if transcript is None:
        return 1.0
    by_id = transcript.by_id()
    ok = sum(
        1
        for r in refs
        if r.segment_id in by_id and quote_matches(r.quote, by_id[r.segment_id].text, QUOTE_RATIO)
    )
    return ok / len(refs)


def _trap_resisted(trap: dict, pred: Minutes, body_items: list[dict]) -> bool:
    kind = trap["kind"]
    if kind == "wrong_number":
        value = _norm_num(trap["value"])
        present = value in {_norm_num(n) for n in _NUMBER.findall(normalise_ar(_body_text(pred)))}
        return not present or value in _norm_num(_flag_text(pred, "number_to_verify"))
    if kind == "unknown_name":
        name = trap["name"].lower()
        present = name in _body_text(pred).lower() or any(
            name in (a.owner or "").lower() for a in pred.actions
        )
        return not present or name in _flag_text(pred, "name_to_verify").lower()
    in_body = any(_ratio(trap["text"], it["text"]) >= MATCH_RATIO for it in body_items)
    if kind == "reversed_decision" and in_body:
        return _ratio(trap["text"], _flag_text(pred, "contradiction")) >= MATCH_RATIO
    return not in_body


def score(
    pred: Minutes,
    gold: dict[str, Any],
    transcript: Transcript | None = None,
    *,
    roster: list[Attendee] | None = None,
    name: str = "",
    language: str = "en",
) -> dict[str, Any]:
    """Per-meeting metrics for ``pred`` against ``gold`` (see module docstring). Metrics whose
    denominator is zero are ``None`` and ignored by ``aggregate``."""
    p_dec_all, p_act = _pred_items(pred)
    contradicted = _contradicted(pred)
    p_dec = [d for i, d in enumerate(p_dec_all) if i not in contradicted]
    g_dec = [{"text": d["statement"], "refs": d.get("refs", [])} for d in gold.get("decisions", [])]
    g_act = [{"text": a["description"], **a} for a in gold.get("actions", [])]
    dec_pairs, act_pairs = _match(g_dec, p_dec), _match(g_act, p_act)
    dp, dr, _ = _prf(len(dec_pairs), len(p_dec) - len(dec_pairs), len(g_dec) - len(dec_pairs))
    ap, ar, af = _prf(len(act_pairs), len(p_act) - len(act_pairs), len(g_act) - len(act_pairs))
    owners = [
        (g_act[gi], p_act[pi])
        for gi, pi in act_pairs
        if g_act[gi].get("owner_confidence") == "explicit" and g_act[gi].get("owner")
    ]
    owner_ok = [(p["owner"] or "").strip().lower() == g["owner"].strip().lower() for g, p in owners]
    dues = [(g_act[gi], p_act[pi]) for gi, pi in act_pairs if g_act[gi].get("due_date")]
    due_ok = [
        p["due_date"] == g["due_date"]
        or (bool(p["due_text"]) and _ratio(p["due_text"], g.get("due_text") or "") >= 80)
        for g, p in dues
    ]
    traps = gold.get("traps", [])
    resisted = [_trap_resisted(t, pred, p_dec_all + p_act) for t in traps]
    body_total = len(pred.decisions) + len(pred.actions) + len(pred.risks)
    unsupported = sum(1 for it in (*pred.decisions, *pred.actions, *pred.risks) if not it.refs)
    by_id = transcript.by_id() if transcript else None
    uncited = sum(
        1
        for d in pred.decisions
        if not d.refs or (by_id is not None and not any(r.segment_id in by_id for r in d.refs))
    )
    return {
        "name": name,
        "language": language,
        "schema_ok": True,
        "decision_precision": dp,
        "decision_recall": dr,
        "decision_tp": len(dec_pairs),
        "decision_fp": len(p_dec) - len(dec_pairs),
        "decision_fn": len(g_dec) - len(dec_pairs),
        "decisions_contradicted": len(contradicted),
        "action_precision": ap,
        "action_recall": ar,
        "action_f1": af,
        "owner_accuracy": sum(owner_ok) / len(owner_ok) if owner_ok else None,
        "due_accuracy": sum(due_ok) / len(due_ok) if due_ok else None,
        "trap_resistance": sum(resisted) / len(resisted) if resisted else None,
        "traps_total": len(traps),
        "traps_resisted": int(sum(resisted)),
        "citation_validity": _citation_validity(pred, transcript),
        "numbers_present_or_flagged": _numbers_metric(pred, transcript),
        "names_resolved_or_flagged": _names_metric(pred, roster or pred.attendees, gold),
        "unsupported_claim_rate": unsupported / body_total if body_total else 0.0,
        "uncited_decisions_in_body": uncited,
    }


# --------------------------------------------------------------------------- gate


def language_gap_applicable(report: EvalReport) -> bool:
    """True when the evaluated set holds at least one non-English (Arabic) meeting, so the
    Arabic/English decision-recall gap is a gate criterion."""
    return any(m.language != "en" for m in report.meetings)


def gate(report: EvalReport, thresholds: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """Apply the rollout thresholds; returns ``(passed, reasons)``.

    Same criteria as ``eval.report.gate``, except that the language-gap criterion is skipped
    when ``language_gap_applicable`` is false (English-only golden set). A ``None`` metric still
    fails its gate, so an empty evaluation never passes.
    """
    passed, reasons = _report_gate(report, thresholds)
    if language_gap_applicable(report):
        return passed, reasons
    kept = [r for r in reasons if not r.startswith(f"{GAP_KEY} ")]
    if len(kept) != len(reasons):
        log.info("eval.gate.language_gap_skipped", reason="no_arabic_meeting")
    return not kept, kept


def write_report(report: EvalReport, path: Path) -> Path:
    """Write ``eval_report.md`` with the gate result of ``gate`` above; for an English-only set
    the language-gap row reads "not applicable" instead of a failing ``n/a``."""
    path = _report_write_report(report, path)
    if language_gap_applicable(report):
        return path
    lines = path.read_text(encoding="utf-8").splitlines()
    table = next(i for i, line in enumerate(lines) if line.startswith("| Metric |"))
    passed, reasons = gate(report)
    head = ["# Praktika evaluation report", "", f"Gate: {'PASS' if passed else 'FAIL'}", ""]
    if reasons:
        head += [f"- {r}" for r in reasons] + [""]
    head += [f"Skipped: {GAP_KEY} is {GAP_NOT_APPLICABLE}.", ""]
    body = [
        f"| {GAP_KEY} (points) | {GAP_NOT_APPLICABLE} |"
        if line.startswith(f"| {GAP_KEY} (points) |")
        else line
        for line in lines[table:]
    ]
    path.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    return path
