"""Domain model contracts: audit hashing, Minutes flags and schema flatness."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from helpers_foundation import (
    NOW,
    audit_event,
    contains_key,
    inline_refs,
    minutes,
)
from pydantic import BaseModel

import praktika.models as models_pkg
from praktika.models import (
    AuditEvent,
    ChunkFindings,
    ClassificationSuggestion,
    Flag,
    MancomMinutes,
    Meeting,
    MergedFindings,
    Minutes,
    Narrative,
    OneToOneMinutes,
    RetractionVerdict,
    Transcript,
)

# --------------------------------------------------------------------------- AuditEvent


def test_audit_hash_stable_and_excludes_hash_field() -> None:
    e1, e2 = audit_event(), audit_event()
    assert e1.compute_hash() == e2.compute_hash() and len(e1.hash) == 0
    sealed = e1.sealed()
    assert sealed.hash == e1.compute_hash() and sealed.verify_hash()
    assert sealed.compute_hash() == e1.compute_hash(), "hash field must not feed the hash"
    assert not e1.verify_hash(), "an unsealed event never verifies"


@pytest.mark.parametrize(
    "change",
    [
        {"actor": "someone-else"},
        {"prev_hash": "1" * 64},
        {"detail": {"method": "chat", "n": 3}},
        {"ts": NOW.replace(second=1)},
        {"model": "llama3.1:8b"},
    ],
)
def test_audit_hash_changes_with_any_field(change: dict[str, Any]) -> None:
    base = audit_event().sealed()
    tampered = base.model_copy(update=change)
    assert tampered.compute_hash() != base.hash
    assert not tampered.verify_hash()


def test_audit_hash_is_canonical_json_sha256() -> None:
    import hashlib

    e = audit_event(detail={"z": 1, "a": [1, 2]})
    canon = json.dumps(
        e.model_dump(mode="json", exclude={"hash"}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert e.compute_hash() == hashlib.sha256(canon.encode()).hexdigest()
    assert e.model_dump(mode="json")["ts"] == "2026-09-16T09:00:00+03:00"


# --------------------------------------------------------------------------- Minutes


def test_blocking_flags_priority_one_uncleared_only() -> None:
    blocking = Flag(
        kind="uncited_item_removed", detail="removed decision", item_json="{}", priority=1
    )
    cleared = Flag(
        kind="identifier_detected",
        detail="IBAN",
        priority=1,
        cleared_by="F. Khalid",
        cleared_at=NOW,
    )
    advisory = Flag(kind="name_to_verify", detail="Tom", priority=2)
    m = minutes(flags=[advisory, cleared, blocking])
    assert m.blocking_flags() == [blocking]
    assert minutes(flags=[advisory, cleared]).blocking_flags() == []
    assert minutes().review.status == "draft" and minutes().version == 1


def test_minutes_subclasses_keep_base_fields() -> None:
    m = OneToOneMinutes(**minutes().model_dump())
    assert m.private is True and m.my_commitments == [] and m.blocking_flags() == []
    mc = MancomMinutes(**minutes().model_dump())
    assert mc.agenda == [] and mc.decisions[0].id == "D1"


# --------------------------------------------------------------------------- schema flatness


LLM_OUTPUT_MODELS = [
    ChunkFindings,
    MergedFindings,
    Narrative,
    RetractionVerdict,
    ClassificationSuggestion,
]


@pytest.mark.parametrize("model", LLM_OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_llm_schemas_are_flat_and_enum_only(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    assert schema["title"] == model.__name__
    flat = inline_refs(schema)
    assert not contains_key(flat, "$ref") and "$defs" not in flat
    assert not contains_key(flat, "$defs")
    assert flat.get("additionalProperties") is False

    def check_enums(node: Any) -> None:
        if isinstance(node, dict):
            if "enum" in node:
                assert all(isinstance(v, str) for v in node["enum"]), node
            for v in node.values():
                check_enums(v)
        elif isinstance(node, list):
            for v in node:
                check_enums(v)

    def is_bounded(prop: dict[str, Any]) -> bool:
        """A direct string property must carry maxLength, enum or const (list items may not)."""
        variants = prop.get("anyOf", [prop])
        return all(
            v.get("type") != "string" or {"maxLength", "enum", "const"} & set(v) for v in variants
        )

    check_enums(flat["properties"])
    for name, prop in flat["properties"].items():
        assert is_bounded(prop), f"unbounded string {model.__name__}.{name}: {prop}"


def test_draft_optional_text_is_plain_string_not_nullable() -> None:
    """Constrained decoding on small local models picks ``null`` for ``anyOf`` fields almost
    every time (qwen2.5:14b returned owner/due_text null for stated values), so optional draft
    text is a required plain string ("" when absent) and ``None`` from playbacks is normalised."""
    from praktika.models import ActionDraft, DecisionDraft, QuestionDraft, RiskDraft

    schema = inline_refs(ChunkFindings.model_json_schema())
    action = schema["properties"]["actions"]["items"]["properties"]
    for field in ("owner", "due_text"):
        assert action[field]["type"] == "string" and "anyOf" not in action[field], field
    a = ActionDraft(
        description="x",
        owner=None,
        owner_confidence="unknown",
        due_text=None,
        refs=["S0001"],
        quote="q",
    )
    assert a.owner == "" and a.due_text == ""
    assert (
        RiskDraft(description="r", severity="low", owner=None, mitigation=None, refs=[]).owner == ""
    )
    assert QuestionDraft(question="q", raised_by=None, owner=None, refs=[]).raised_by == ""
    d = DecisionDraft(
        statement="s",
        kind="noted",
        decided_by="Chair",
        dissent_or_conditions=None,
        refs=["S0001"],
        quote="q",
    )
    assert d.dissent_or_conditions == ""


@pytest.mark.parametrize(
    "model",
    [Minutes, MancomMinutes, OneToOneMinutes, Meeting, Transcript, AuditEvent],
    ids=lambda m: m.__name__,
)
def test_domain_schemas_inline_without_recursion(model: type[BaseModel]) -> None:
    flat = inline_refs(model.model_json_schema())
    assert not contains_key(flat, "$ref")


def test_inline_refs_detects_recursion() -> None:
    recursive = {
        "$defs": {"N": {"type": "object", "properties": {"child": {"$ref": "#/$defs/N"}}}},
        "properties": {"root": {"$ref": "#/$defs/N"}},
    }
    with pytest.raises(RecursionError):
        inline_refs(recursive)


def test_no_voice_or_analytics_fields_in_models() -> None:
    forbidden = {"embedding", "voiceprint", "talk_time", "sentiment", "attendance_score"}
    for name in models_pkg.__all__:
        cls = getattr(models_pkg, name)
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            assert not forbidden & set(cls.model_fields), name


def test_datetimes_survive_with_timezone() -> None:
    e = audit_event(ts=datetime(2026, 9, 16, 6, 0, tzinfo=UTC))
    back = AuditEvent.model_validate_json(e.model_dump_json())
    assert back.ts == e.ts and back.ts.utcoffset() is not None
