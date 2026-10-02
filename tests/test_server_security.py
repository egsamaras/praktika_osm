"""Review API security regressions: reviewer text never carries identifiers to the model or
the record (C-06), denials and read-only accesses are audited (docs/DEPLOYMENT.md, "Audit
events"), and a rejected deferred decision leaves the follow-ups (C-07)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import helpers_foundation
import numpy as np
import pytest
import soundfile as sf
from conftest import FakeIdentity, FakeLLM, make_transcript
from fastapi.testclient import TestClient
from helpers_foundation import meeting as base_meeting
from helpers_foundation import minutes as base_minutes
from helpers_foundation import ref

from praktika import server
from praktika.audit import AuditLog, JsonlAuditSink
from praktika.config import Settings
from praktika.models import Decision, MeetingState
from praktika.store.repo import SqliteStore

NOW = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)
MID = "M-20260916-a1b2"
IBAN = "GB82WEST12345698765432"
CARD = "4111 1111 1111 1111"
PHONE = "+973 3000 0123"  # Bahrain's 30xx mobile block is unallocated: no subscriber has it
HEADERS = {"X-Praktika-Review": "1"}


def name_stored_transcript(monkeypatch: pytest.MonkeyPatch, transcript: Any) -> None:
    """Drafts built by the shared helper name ``transcript`` as their source. A draft is
    approvable only when the transcript it names is stored (``stale_draft_reason``), and the
    helper's placeholder hash names none."""
    placeholder = helpers_foundation.provenance
    monkeypatch.setattr(
        helpers_foundation,
        "provenance",
        lambda: placeholder().model_copy(update={"transcript_sha256": transcript.sha256()}),
    )


@pytest.fixture
def env(tmp_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store = SqliteStore(tmp_path / "praktika.db")
    audit = AuditLog(JsonlAuditSink(tmp_settings.data_dir / "audit.jsonl"), store, None)
    store.save_meeting(base_meeting().model_copy(update={"state": MeetingState.draft_ready}))
    transcript = make_transcript("en", n=12)
    store.save_transcript(transcript, delete_after=None)
    name_stored_transcript(monkeypatch, transcript)
    store.save_minutes(base_minutes())
    chain = tmp_settings.data_dir / "audit.jsonl"

    def events() -> list[dict[str, Any]]:
        if not chain.exists():
            return []
        return [json.loads(line) for line in chain.read_text().splitlines()]

    def client(identity: Any, llm: FakeLLM | None = None) -> TestClient:
        app = server.create_app(
            tmp_settings, store, identity, audit, llm_client=llm, clock=lambda: NOW
        )
        return TestClient(app, base_url="http://127.0.0.1", headers=HEADERS)

    return {"store": store, "settings": tmp_settings, "events": events, "client": client}


ORGANISER = FakeIdentity(source="session")


# --------------------------------------------------------------------------- C-06


@pytest.mark.parametrize(
    "instruction",
    [
        f"use account {IBAN} for the vendor",
        f"the card was {CARD}, mention it",
        f"call Omar on {PHONE} about this",
        "note that Karim Mansour is paid 4,500 dinars a month",
    ],
)
def test_regenerate_instruction_with_identifier_is_refused(
    env: dict[str, Any], instruction: str
) -> None:
    fake = FakeLLM()
    client = env["client"](ORGANISER, fake)
    r = client.post(
        f"/api/minutes/{MID}/regenerate", json={"section": "summary", "instruction": instruction}
    )
    assert r.status_code == 422, r.text
    assert "identifier" in r.json()["detail"]
    for raw in (IBAN, CARD, PHONE, "4,500"):
        assert raw not in r.text, "the response never echoes the value"
    assert fake.calls == [], "nothing reached the model"
    assert env["store"].latest_minutes(MID).version == 1
    assert not any(e["event"] == "llm.call" for e in env["events"]())
    # a clean instruction still regenerates
    r = client.post(
        f"/api/minutes/{MID}/regenerate",
        json={"section": "summary", "instruction": "Shorter, and lead with the pilot decision."},
    )
    assert r.status_code == 200, r.text
    assert fake.calls and "Shorter, and lead" in fake.calls[0].user


def test_modified_item_with_identifier_never_reaches_minutes_or_model(
    env: dict[str, Any],
) -> None:
    fake = FakeLLM()
    client = env["client"](ORGANISER, fake)
    body = {"action": "modify", "reason_code": "wording", "after": f"Pay from {IBAN} by Thursday"}
    r = client.post(f"/api/minutes/{MID}/items/A1", json=body)
    assert r.status_code == 422 and "IBAN" in r.json()["detail"] and IBAN not in r.text
    minutes = env["store"].latest_minutes(MID)
    assert minutes.actions[0].description == "Draft the notice"
    assert minutes.review.items == [] and env["store"].list_review_items(MID, 1) == []
    # a clean edit is accepted
    r = client.post(
        f"/api/minutes/{MID}/items/A1",
        json={"action": "modify", "reason_code": "wording", "after": "Draft the privacy notice"},
    )
    assert r.status_code == 200, r.text
    # nothing in the stored minutes or in a later narrative regeneration carries the value
    r = client.post(
        f"/api/minutes/{MID}/regenerate",
        json={"section": "summary", "instruction": "Tighten the summary."},
    )
    assert r.status_code == 200, r.text
    assert fake.calls and "Draft the privacy notice" in fake.calls[0].user, "edits reach the model"
    assert all(IBAN not in c.user and IBAN not in c.system for c in fake.calls)
    assert IBAN not in env["store"].latest_minutes(MID).model_dump_json()


# --------------------------------------------------------------------------- audit trail


def _denied(events: list[dict[str, Any]]) -> list[tuple[str | None, str, str, int]]:
    return [
        (e["meeting_id"], e["detail"]["reason"], e["detail"]["route"], e["detail"]["status"])
        for e in events
        if e["event"] == "auth.denied"
    ]


def test_denials_are_audited(env: dict[str, Any], tmp_path: Path) -> None:
    nobody = FakeIdentity(user="c.tractor@acme.test", source="oidc", groups=())
    client = env["client"](nobody)
    assert client.get("/api/meetings").status_code == 401
    assert _denied(env["events"]()) == [(None, "unauthenticated", "GET /api/meetings", 401)]

    other = FakeIdentity(user="r.haddad@acme.test", source="oidc", groups=("Praktika-Users",))
    client = env["client"](other)
    slice_ = {"start": 0, "end": 5}
    assert client.get(f"/api/meetings/{MID}").status_code == 403
    assert client.get(f"/api/meetings/{MID}/audio", params=slice_).status_code == 403
    assert client.get(f"/api/export/{MID}.md").status_code == 403
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 403
    denied = [e for e in env["events"]() if e["event"] == "auth.denied"][1:]
    assert [(e["meeting_id"], e["detail"]["reason"]) for e in denied] == [
        (MID, "not_visible"),
        (MID, "not_visible"),
        (MID, "not_visible"),
        (MID, "not_visible"),
    ]
    assert all(e["actor"] == "r.haddad@acme.test" and e["actor_source"] == "oidc"
               for e in denied)  # fmt: skip
    assert [e["detail"]["route"] for e in denied] == [
        f"GET /api/meetings/{MID}",
        f"GET /api/meetings/{MID}/audio",
        f"GET /api/export/{MID}.md",
        f"POST /api/minutes/{MID}/approve",
    ]

    dpo = FakeIdentity(user="d.po@acme.test", source="oidc", groups=("Praktika-DPO",))
    client = env["client"](dpo)
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 403
    last = [e for e in env["events"]() if e["event"] == "auth.denied"][-1]
    assert last["actor"] == "d.po@acme.test" and last["detail"]["reason"] == "read_only_role"
    assert env["store"].get_meeting(MID).state is MeetingState.draft_ready


def test_read_only_access_is_audited(env: dict[str, Any], tmp_path: Path) -> None:
    store, events = env["store"], env["events"]
    wav = tmp_path / "system.wav"
    t = np.arange(16000 * 2) / 16000
    sf.write(str(wav), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    store.save_media(MID, wav, "ab" * 32, kind="audio", delete_after=None)
    dpo = FakeIdentity(user="d.po@acme.test", source="oidc", groups=("Praktika-DPO",))
    client = env["client"](dpo)
    assert client.get(f"/api/meetings/{MID}").status_code == 200
    assert store.get_meeting(MID).state is MeetingState.draft_ready, "a read moves nothing"
    r = client.get(f"/api/meetings/{MID}/audio", params={"start": 0.5, "end": 1.5})
    assert r.status_code == 206
    opened = [e for e in events() if e["event"] == "review.opened"]
    assert len(opened) == 1 and opened[0]["detail"]["read_only"] is True
    assert opened[0]["actor"] == "d.po@acme.test" and opened[0]["detail"]["version"] == 1
    read = [e for e in events() if e["event"] == "audio.read"]
    assert len(read) == 1 and read[0]["detail"] == {
        "start": 0.5, "end": 1.5, "read_only": True
    }  # fmt: skip
    assert read[0]["object"] == "system.wav" and read[0]["classification"] == "internal"
    # the organiser's own read is the usual review.opened (not read-only) plus an audio.read
    client = env["client"](ORGANISER)
    assert client.get(f"/api/meetings/{MID}").status_code == 200
    r = client.get(f"/api/meetings/{MID}/audio", params={"start": 0, "end": 1})
    assert r.status_code == 206
    opened = [e for e in events() if e["event"] == "review.opened"]
    assert [o["detail"].get("read_only", False) for o in opened] == [True, False]
    assert [e["detail"]["read_only"] for e in events() if e["event"] == "audio.read"] == [
        True,
        False,
    ]


# --------------------------------------------------------------------------- C-07


def test_rejecting_a_deferred_decision_drops_its_follow_up(env: dict[str, Any]) -> None:
    store = env["store"]
    deferred = Decision(
        id="D2",
        statement="Retention question deferred to next week",
        kind="deferred",
        decided_by="Chair",
        refs=[ref("S0010")],
    )
    minutes = base_minutes(decisions=[*base_minutes().decisions, deferred])
    store.set_review_status(minutes)
    client = env["client"](ORGANISER)
    detail = client.get(f"/api/meetings/{MID}").json()
    assert detail["minutes"]["follow_ups"] == ["Retention question"], "stored as-is until edited"
    r = client.post(
        f"/api/minutes/{MID}/items/D1", json={"action": "accept", "reason_code": "accurate"}
    )
    assert r.status_code == 200
    assert r.json()["minutes"]["follow_ups"] == [deferred.statement], "derived from the body"
    r = client.post(
        f"/api/minutes/{MID}/items/D2", json={"action": "reject", "reason_code": "not_said"}
    )
    assert r.status_code == 200
    assert r.json()["minutes"]["follow_ups"] == []
    stored = store.latest_minutes(MID)
    assert stored.follow_ups == [] and [d.id for d in stored.decisions] == ["D1"]
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 200
    text = client.get(f"/api/export/{MID}.md").text
    assert "Retention question" not in text


def test_session_token_persists_per_data_dir(tmp_path) -> None:
    """The review link printed by start/ingest/generate must open directly: the local-mode
    token is stored once per data directory (0600) and reused by every serve run."""
    import os

    from praktika.server_auth import TOKEN_FILE, session_token_for

    first = session_token_for(tmp_path)
    second = session_token_for(tmp_path)
    assert first == second and len(first) >= 24
    mode = os.stat(tmp_path / TOKEN_FILE).st_mode & 0o777
    assert mode == 0o600
    assert session_token_for(tmp_path / "other") != first


def test_speaker_name_with_an_identifier_is_refused(env: dict[str, Any]) -> None:
    """A speaker name is reviewer-typed text stored in the transcript and rendered into every
    later model call, so it is held to the same rule as an edited item (C-06)."""
    client = env["client"](ORGANISER)
    bad = client.post(
        f"/api/meetings/{MID}/speakers",
        json={"SPEAKER_01": "R. Haddad (a/c 12345678901)"},
        headers=HEADERS,
    )
    assert bad.status_code == 422 and "ACC" in bad.text and "12345678901" not in bad.text
    ok = client.post(
        f"/api/meetings/{MID}/speakers", json={"SPEAKER_01": "R. Haddad"}, headers=HEADERS
    )
    assert ok.status_code == 200


# --------------------------------------------------------------------------- stale drafts


def test_review_page_refuses_a_draft_awaiting_regeneration(env: dict[str, Any]) -> None:
    """The review page applies ``praktika approve``'s rule: a meeting whose new transcript awaits
    ``praktika generate`` (state ``transcribing``) cannot have its old draft approved."""
    store: SqliteStore = env["store"]
    store.set_state(MID, MeetingState.transcribing)
    client = env["client"](ORGANISER)
    r = client.post(f"/api/minutes/{MID}/approve", json={"reason_code": "accurate"})
    assert r.status_code == 409 and "praktika generate" in r.json()["detail"]
    assert store.get_meeting(MID).state is MeetingState.transcribing  # type: ignore[union-attr]
    assert not [e for e in env["events"]() if e["event"] == "review.approved"]


def test_review_page_refuses_a_draft_of_an_older_transcript(env: dict[str, Any]) -> None:
    """A newer transcript with different words makes the stored draft stale, whatever the state."""
    store: SqliteStore = env["store"]
    first = store.get_transcript(MID)
    draft = store.latest_minutes(MID)
    provenance = draft.provenance.model_copy(update={"transcript_sha256": first.sha256()})
    store.save_minutes(
        draft.model_copy(update={"version": draft.version + 1, "provenance": provenance})
    )
    newer = make_transcript("en", n=12)
    newer.segments[0] = newer.segments[0].model_copy(update={"text": "Something else was said."})
    store.save_transcript(newer, delete_after=None)
    client = env["client"](ORGANISER)
    r = client.post(f"/api/minutes/{MID}/approve", json={"reason_code": "accurate"})
    assert r.status_code == 409 and "older transcript" in r.json()["detail"]
