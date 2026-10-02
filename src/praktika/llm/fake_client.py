"""Offline ``LLMClient`` for ``llm_provider="fake"`` (no Ollama, no network, no weights).

Contract: ``FakeLLM.complete_json`` returns a schema-valid object for every schema the pipeline
asks for (``ChunkFindings``, ``MergedFindings``, ``Narrative``, ``RetractionVerdict``,
``ClassificationSuggestion``), derived deterministically from the transcript lines in the user
message: one decision citing the first segment with a verbatim quote, one action citing the
second, one key point per early segment. Quotes are verbatim so the verifier keeps the items;
statements reuse transcript wording so the number check never flags a figure the fake invented.
It is a placeholder for exercising the pipeline, review UI and exports end to end, not a
summariser: the minutes it drafts say so in their summary.
"""

from __future__ import annotations

import json
import re
from typing import Any

from praktika.errors import LLMError
from praktika.logging import get_logger

log = get_logger(__name__)

NAME = "fake"
_LINE_RE = re.compile(r"^\[(S\d{4,5}) \d\d:\d\d:\d\d-\d\d:\d\d:\d\d ([^|\]]*)\|(\w+)\] (.*)$", re.M)
_PLACEHOLDER = (
    "Placeholder minutes drafted by the offline fake model (PRAKTIKA_LLM_PROVIDER=fake): "
    "the items below quote the transcript verbatim and carry no judgement."
)
_QUOTE_MAX = 240
_STATEMENT_MAX = 400
_KEY_POINTS = 5


def transcript_lines(user: str) -> list[tuple[str, str, str, str]]:
    """``(segment_id, speaker, language, text)`` for every rendered transcript line in ``user``."""
    return [
        (m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip())
        for m in _LINE_RE.finditer(user)
    ]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _quote(text: str) -> str:
    """A verbatim prefix of ``text`` on a word boundary, within the ``quote`` length limit."""
    if len(text) <= _QUOTE_MAX:
        return text
    cut = text[:_QUOTE_MAX]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def _json_payload(user: str) -> Any:
    """The JSON block a reduce/narrative prompt embeds, or ``None`` when there is none."""
    for opener in ("[", "{"):
        start = user.find(opener)
        if start == -1:
            continue
        end = max(user.rfind("]"), user.rfind("}"))
        try:
            return json.loads(user[start : end + 1])
        except ValueError:
            continue
    return None


def findings_from_lines(lines: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """A ``ChunkFindings`` object: decision from line 1, action from line 2, early key points."""
    out: dict[str, Any] = {
        "decisions": [],
        "actions": [],
        "questions": [],
        "risks": [],
        "key_points": [],
        "figures": [],
    }
    if not lines:
        return out
    sid, speaker, _, text = lines[0]
    out["decisions"].append(
        {
            "statement": _clip(f"Noted (placeholder): {text}", _STATEMENT_MAX),
            "kind": "noted",
            "decided_by": speaker or "unknown",
            "dissent_or_conditions": None,
            "refs": [sid],
            "quote": _quote(text),
        }
    )
    if len(lines) > 1:
        sid, speaker, _, text = lines[1]
        known = bool(speaker) and not speaker.upper().startswith(("SPEAKER_", "ROOM", "UNKNOWN"))
        out["actions"].append(
            {
                "description": _clip(f"Follow up (placeholder): {text}", _STATEMENT_MAX),
                "owner": speaker if known else None,
                "owner_confidence": "inferred" if known else "unknown",
                "due_text": None,
                "refs": [sid],
                "quote": _quote(text),
            }
        )
    for sid, _, _, text in lines[:_KEY_POINTS]:
        out["key_points"].append({"topic": "Transcript", "point": _clip(text, 300), "refs": [sid]})
    return out


def merge_findings(payload: Any) -> dict[str, Any]:
    """Concatenate per-chunk findings in order (the reduce stage) without inventing anything."""
    merged: dict[str, Any] = {
        "decisions": [],
        "actions": [],
        "questions": [],
        "risks": [],
        "key_points": [],
        "figures": [],
        "retracted_decisions": [],
    }
    for chunk in payload if isinstance(payload, list) else []:
        if isinstance(chunk, dict):
            for key in ("decisions", "actions", "questions", "risks", "key_points", "figures"):
                merged[key].extend(chunk.get(key, []))
    return merged


def narrative_from(payload: Any) -> dict[str, Any]:
    """A ``Narrative`` whose topics reuse the merged key points and their segment ids."""
    points = payload.get("key_points", []) if isinstance(payload, dict) else []
    refs = sorted({r for p in points for r in p.get("refs", [])})[:12]
    topic = {
        "title": "Transcript (placeholder)",
        "summary": "Key points quoted verbatim from the transcript by the offline fake model.",
        "key_points": [p["point"] for p in points[:8]],
        "refs": refs,
    }
    return {"summary": _PLACEHOLDER, "topics": [topic] if points else []}


class FakeLLM:
    """Deterministic offline stand-in for Ollama / vLLM; see the module docstring."""

    name = NAME

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        seed: int = 7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        self.calls += 1
        title = str(schema.get("title", ""))
        log.debug("llm.fake_call", schema=title, user_chars=len(user))
        if title == "ChunkFindings":
            return findings_from_lines(transcript_lines(user))
        if title == "MergedFindings":
            return merge_findings(_json_payload(user))
        if title == "Narrative":
            return narrative_from(_json_payload(user))
        if title == "RetractionVerdict":
            return {"retracted": False, "refs": [], "note": "Placeholder: no reversal checked."}
        if title == "ClassificationSuggestion":
            return {
                "suggested": "internal",
                "reasons": ["placeholder"],
                "identifiers_seen": [],
                "mnpi_keywords": [],
            }
        raise LLMError(f"FakeLLM has no response for schema {title!r}")

    def model_digest(self) -> str:
        return "sha256:fake"

    def with_model(self, model: str) -> FakeLLM:
        """The fake has one behaviour whatever the model name; returns itself."""
        return self
