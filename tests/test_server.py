"""Review API: state machine, HITL gates, audio, export."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
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
from praktika.errors import PraktikaError
from praktika.identity import IdentityError
from praktika.models import Classification, Decision, Flag, MeetingState, Segment
from praktika.store.repo import SqliteStore

NOW = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)
MID = "M-20260916-a1b2"
LOCAL = "http://127.0.0.1"
REVIEW_HEADERS = {"X-Praktika-Review": "1"}


def client_for(app: Any, *, token: str | None = None, **headers: str) -> TestClient:
    """A browser-like client: loopback ``Host``, the CSRF header on every request and, when
    the app was built with a session token, that token."""
    extra = dict(REVIEW_HEADERS, **headers)
    if token:
        extra["X-Praktika-Token"] = token
    return TestClient(app, base_url=LOCAL, headers=extra)


class _Refusing:
    """An identity provider that rejects every request (OIDC without a bearer token)."""

    def current(self, request: Any = None) -> Any:
        raise IdentityError("missing bearer token")


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
    """Store + audit + app around one draft-ready meeting with a transcript and minutes."""
    store = SqliteStore(tmp_path / "praktika.db")
    identity = FakeIdentity(source="session")
    audit = AuditLog(JsonlAuditSink(tmp_settings.data_dir / "audit.jsonl"), store, identity)
    meeting = base_meeting().model_copy(update={"state": MeetingState.draft_ready})
    store.save_meeting(meeting)
    transcript = make_transcript("mixed", n=12)
    store.save_transcript(transcript, delete_after=None)
    name_stored_transcript(monkeypatch, transcript)
    store.save_minutes(base_minutes())
    app = server.create_app(tmp_settings, store, identity, audit, clock=lambda: NOW)
    chain = tmp_settings.data_dir / "audit.jsonl"

    def events() -> list[dict[str, Any]]:
        if not chain.exists():
            return []
        return [json.loads(line) for line in chain.read_text().splitlines()]

    return {
        "store": store,
        "audit": audit,
        "settings": tmp_settings,
        "identity": identity,
        "client": client_for(app),
        "transcript": transcript,
        "events": events,
    }


def _approve(client: TestClient, reason: str = "accurate") -> Any:
    return client.post(f"/api/minutes/{MID}/approve", json={"reason_code": reason})


def _write_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)


# --------------------------------------------------------------------------- binding


def test_local_mode_binds_loopback_only(tmp_settings: Settings) -> None:
    assert server.bind_address(tmp_settings) == ("127.0.0.1", 8793)
    exposed = tmp_settings.model_copy(update={"review_host": "0.0.0.0"})  # noqa: S104
    with pytest.raises(PraktikaError, match="service"):
        server.bind_address(exposed)
    with pytest.raises(PraktikaError):
        server.create_app(exposed, SqliteStore(":memory:"), FakeIdentity(), None)  # type: ignore[arg-type]
    service = tmp_settings.model_copy(
        update={"review_host": "0.0.0.0", "mode": "service", "identity_provider": "oidc"}  # noqa: S104
    )
    assert server.bind_address(service) == ("0.0.0.0", 8793)  # noqa: S104


@pytest.mark.parametrize("provider", ["session", "fake"])
def test_service_mode_network_bind_needs_oidc(tmp_settings: Settings, provider: str) -> None:
    """A network-reachable bind with a provider that authenticates nobody is refused."""
    exposed = tmp_settings.model_copy(
        update={"review_host": "0.0.0.0", "mode": "service", "identity_provider": provider}  # noqa: S104
    )
    with pytest.raises(PraktikaError, match="oidc"):
        server.bind_address(exposed)
    with pytest.raises(PraktikaError, match="oidc"):
        server.create_app(exposed, SqliteStore(":memory:"), FakeIdentity(), None)  # type: ignore[arg-type]
    # loopback in service mode is refused too: behind the gateway (host networking, a sidecar
    # forwarding to 127.0.0.1) every network caller would be the console user
    local = exposed.model_copy(update={"review_host": "127.0.0.1"})
    with pytest.raises(PraktikaError, match="service mode"):
        server.bind_address(local)
    with pytest.raises(PraktikaError, match="oidc"):
        server.create_app(local, SqliteStore(":memory:"), FakeIdentity(), None)  # type: ignore[arg-type]
    with pytest.raises(PraktikaError, match="oidc"):
        server.require_service_identity(local)
    server.require_service_identity(local.model_copy(update={"identity_provider": "oidc"}))
    server.require_service_identity(tmp_settings)  # local mode: any provider


def test_health_needs_no_identity(tmp_settings: Settings, tmp_path: Path) -> None:
    """The container HEALTHCHECK must answer without a bearer token and without data."""
    store = SqliteStore(tmp_path / "praktika.db")
    audit = AuditLog(JsonlAuditSink(tmp_settings.data_dir / "audit.jsonl"), store, None)
    client = client_for(server.create_app(tmp_settings, store, _Refusing(), audit))
    assert client.get("/api/health").json() == {"status": "ok", "mode": "local"}
    assert client.get("/api/meetings").status_code == 401


def test_identity_failure_is_401(env: dict[str, Any]) -> None:
    app = server.create_app(env["settings"], env["store"], _Refusing(), env["audit"])
    assert client_for(app).get("/api/meetings").status_code == 401


# --------------------------------------------------------------------------- authorisation


def _app_for(env: dict[str, Any], identity: Any) -> TestClient:
    return client_for(
        server.create_app(env["settings"], env["store"], identity, env["audit"], clock=lambda: NOW)
    )


def test_identity_without_praktika_role_is_401(env: dict[str, Any]) -> None:
    """An Entra user with a valid token but no Praktika-* group (a contractor) gets nothing."""
    nobody = FakeIdentity(user="c.tractor@acme.test", source="oidc", groups=())
    client = _app_for(env, nobody)
    assert client.get("/api/meetings").status_code == 401
    assert client.get(f"/api/meetings/{MID}").status_code == 401
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 401
    assert env["store"].get_meeting(MID).state is MeetingState.draft_ready


def test_other_organiser_is_403_and_list_is_filtered(env: dict[str, Any]) -> None:
    other = FakeIdentity(user="r.haddad@acme.test", source="oidc", groups=("Praktika-Users",))
    client = _app_for(env, other)
    assert client.get("/api/meetings").json() == []
    assert client.get(f"/api/meetings/{MID}").status_code == 403
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 403
    assert client.post(f"/api/minutes/{MID}/flags/0/clear").status_code == 403
    assert client.get(f"/api/export/{MID}.md").status_code == 403
    audio = client.get(f"/api/meetings/{MID}/audio", params={"start": 0, "end": 1})
    assert audio.status_code == 403
    assert env["store"].get_meeting(MID).state is MeetingState.draft_ready, "nothing moved"
    assert not any(e["event"] == "review.opened" for e in env["events"]())


def test_secretary_sees_shared_but_not_private_and_dpo_is_read_only(env: dict[str, Any]) -> None:
    store = env["store"]
    private = base_meeting().model_copy(
        update={"id": "M-20260916-0101", "state": MeetingState.draft_ready, "private": True}
    )
    store.save_meeting(private)
    store.save_minutes(base_minutes(meeting_id=private.id))
    secretary = FakeIdentity(user="s.ecretary@acme.test", source="oidc",
                             groups=("Praktika-Secretaries",))  # fmt: skip
    client = _app_for(env, secretary)
    assert [m["id"] for m in client.get("/api/meetings").json()] == [MID]
    assert client.get(f"/api/meetings/{private.id}").status_code == 403
    assert client.post(f"/api/minutes/{private.id}/approve", json={}).status_code == 403
    r = client.post(f"/api/minutes/{MID}/items/D1", json={"action": "accept", "reason_code": "ok"})
    assert r.status_code == 200, "a secretary reviews non-private meetings"

    dpo = FakeIdentity(user="d.po@acme.test", source="oidc", groups=("Praktika-DPO",))
    client = _app_for(env, dpo)
    assert {m["id"] for m in client.get("/api/meetings").json()} == {MID, private.id}
    assert client.get(f"/api/meetings/{private.id}").status_code == 200
    assert store.get_meeting(private.id).state is MeetingState.draft_ready, "read-only: no move"
    assert client.post(f"/api/minutes/{private.id}/approve", json={}).status_code == 403
    assert client.post(f"/api/minutes/{MID}/discard", json={}).status_code == 403


def test_local_console_user_is_the_operator(env: dict[str, Any]) -> None:
    """The laptop's session identity carries no AD groups; as the organiser it may review."""
    local = FakeIdentity(user="f.khalid@acme.test", source="local", groups=())
    client = _app_for(env, local)
    assert [m["id"] for m in client.get("/api/meetings").json()] == [MID]
    assert client.get(f"/api/meetings/{MID}").status_code == 200


# --------------------------------------------------------------------------- host / CSRF


def test_foreign_host_and_origin_refused(env: dict[str, Any]) -> None:
    app = env["client"].app
    rebinding = TestClient(app, base_url="http://evil.example:8793", headers=REVIEW_HEADERS)
    assert rebinding.get("/api/meetings").status_code == 400
    assert rebinding.get("/").status_code == 400
    ok = env["client"]
    assert ok.get("/api/meetings").status_code == 200
    assert (
        TestClient(app, base_url="http://localhost:8793", headers=REVIEW_HEADERS)
        .get("/api/health")
        .status_code
        == 200
    )
    # a cross-site form POST: foreign Origin, no custom header
    r = TestClient(app, base_url=LOCAL).post(
        f"/api/minutes/{MID}/flags/0/clear",
        headers={
            "Origin": "https://evil.example",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert r.status_code == 403
    r = ok.post(f"/api/minutes/{MID}/approve", json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = ok.post(
        f"/api/minutes/{MID}/approve", json={}, headers={"Referer": "https://evil.example/x"}
    )
    assert r.status_code == 403
    r = TestClient(app, base_url=LOCAL).post(f"/api/minutes/{MID}/approve", json={})
    assert r.status_code == 403, "POST without X-Praktika-Review is refused"
    assert env["store"].latest_minutes(MID).review.status == "draft"
    r = ok.post(f"/api/minutes/{MID}/approve", json={}, headers={"Origin": "http://127.0.0.1:8793"})
    assert r.status_code == 200


def test_local_session_token_required(env: dict[str, Any]) -> None:
    token = server.new_session_token()
    app = server.create_app(
        env["settings"], env["store"], env["identity"], env["audit"], session_token=token
    )
    assert client_for(app).get("/api/meetings").status_code == 401
    assert client_for(app, token="wrong").get("/api/meetings").status_code == 401  # noqa: S106
    assert client_for(app).get("/api/health").status_code == 200
    assert client_for(app).get("/").status_code == 200, "the page itself loads; the API does not"
    assert client_for(app, token=token).get("/api/meetings").status_code == 200
    url = server.review_url(env["settings"], token, MID)
    assert url == f"http://127.0.0.1:8793/?t={token}#/meetings/{MID}"


# --------------------------------------------------------------------------- state machine


def test_state_machine_transitions(env: dict[str, Any]) -> None:
    client, store, events = env["client"], env["store"], env["events"]
    listing = client.get("/api/meetings").json()
    assert [m["id"] for m in listing] == [MID] and listing[0]["review_status"] == "draft"

    detail = client.get(f"/api/meetings/{MID}").json()
    assert detail["meeting"]["state"] == "in_review"
    assert store.get_meeting(MID).state is MeetingState.in_review
    assert len(detail["segments"]) == 12 and detail["minutes"]["version"] == 1
    assert any(e["event"] == "review.opened" for e in events())
    # opening again is idempotent: no second review.opened
    client.get(f"/api/meetings/{MID}")
    assert sum(e["event"] == "review.opened" for e in events()) == 1

    r = client.post(f"/api/minutes/{MID}/approve", json={"reason_code": "accurate"})
    assert r.status_code == 200 and r.json()["state"] == "approved"
    stored = store.latest_minutes(MID)
    assert stored.review.status == "approved"
    assert stored.review.reviewer == env["identity"].user
    assert stored.review.reviewer_source == "session"
    assert store.get_meeting(MID).state is MeetingState.approved

    # approved is terminal for review actions
    assert client.post(f"/api/minutes/{MID}/approve", json={}).status_code == 409
    assert client.post(f"/api/minutes/{MID}/discard", json={}).status_code == 409
    assert (
        client.post(
            f"/api/minutes/{MID}/items/D1", json={"action": "accept", "reason_code": "accurate"}
        ).status_code
        == 409
    )

    # a fresh draft can be discarded, after which nothing else moves it
    other = base_meeting().model_copy(
        update={"id": "M-20260916-b2c3", "state": MeetingState.draft_ready}
    )
    store.save_meeting(other)
    store.save_minutes(base_minutes(meeting_id=other.id))
    r = client.post(f"/api/minutes/{other.id}/discard", json={"reason_code": "duplicate"})
    assert r.status_code == 200 and store.get_meeting(other.id).state is MeetingState.discarded
    assert store.latest_minutes(other.id).review.status == "discarded"
    assert client.post(f"/api/minutes/{other.id}/approve", json={}).status_code == 409
    names = [e["event"] for e in events()]
    assert "review.approved" in names and "review.discarded" in names
    assert client.get("/api/meetings/M-20260916-ffff").status_code == 404


# --------------------------------------------------------------------------- flags


def _blocking_flag(item_id: str = "D9") -> Flag:
    removed = Decision(
        id=item_id,
        statement="Vendor contract renewed for two years",
        kind="approved",
        decided_by="Chair",
        refs=[ref("S0003", "Agreed.")],
    )
    return Flag(
        kind="uncited_item_removed",
        priority=1,
        detail=f"{removed.statement} — removed: no verifiable citation",
        item_json=json.dumps(removed.model_dump(mode="json")),
    )


def test_approve_blocked_by_open_priority_one_flag(env: dict[str, Any]) -> None:
    client, store = env["client"], env["store"]
    flags = [Flag(kind="name_to_verify", detail="Karim", priority=2), _blocking_flag()]
    store.set_review_status(base_minutes(flags=flags))

    detail = client.get(f"/api/meetings/{MID}").json()
    assert [f["priority"] for f in detail["flags"]] == [1, 2]  # priority 1 first
    assert detail["flags"][0]["n"] == 1  # original index kept for the clear route

    r = client.post(f"/api/minutes/{MID}/approve", json={"reason_code": "accurate"})
    assert r.status_code == 403 and "priority-1" in r.json()["detail"]
    assert store.get_meeting(MID).state is MeetingState.in_review

    assert client.post(f"/api/minutes/{MID}/flags/7/clear").status_code == 404
    r = client.post(f"/api/minutes/{MID}/flags/1/clear")
    assert r.status_code == 200 and r.json()["blocking"] == 0
    cleared = store.latest_minutes(MID).flags[1]
    assert cleared.cleared_by == env["identity"].user and cleared.cleared_at == NOW
    assert _approve(client).status_code == 200


def test_restore_removed_item(env: dict[str, Any]) -> None:
    client, store = env["client"], env["store"]
    store.set_review_status(base_minutes(flags=[_blocking_flag("D9")]))
    assert (
        client.post(
            f"/api/minutes/{MID}/items/D9", json={"action": "accept", "reason_code": "accurate"}
        ).status_code
        == 404
    )

    r = client.post(
        f"/api/minutes/{MID}/items/D9", json={"action": "restore", "reason_code": "cited_elsewhere"}
    )
    assert r.status_code == 200
    minutes = store.latest_minutes(MID)
    assert [d.id for d in minutes.decisions] == ["D1", "D9"]
    assert minutes.decisions[1].statement == "Vendor contract renewed for two years"
    assert minutes.flags[0].cleared and not minutes.blocking_flags()
    items = store.list_review_items(MID, 1)
    assert [(i.item_id, i.action, i.reason_code) for i in items] == [
        ("D9", "restore", "cited_elsewhere")
    ]
    assert items[0].after == "Vendor contract renewed for two years"
    # restoring twice is refused: the flag is already cleared
    assert (
        client.post(
            f"/api/minutes/{MID}/items/D9", json={"action": "restore", "reason_code": "other"}
        ).status_code
        == 404
    )
    assert _approve(client).status_code == 200


def test_review_items_stored_with_reason_codes(env: dict[str, Any]) -> None:
    client, store, events = env["client"], env["store"], env["events"]
    post = lambda item, body: client.post(f"/api/minutes/{MID}/items/{item}", json=body)  # noqa: E731
    assert post("D1", {"action": "accept", "reason_code": "accurate"}).status_code == 200
    assert post("A1", {"action": "modify", "reason_code": "wording"}).status_code == 422
    r = post(
        "A1",
        {
            "action": "modify",
            "reason_code": "wording",
            "before": "Draft the notice",
            "after": "Draft the privacy notice",
        },
    )
    assert r.status_code == 200
    assert post("Q1", {"action": "reject", "reason_code": "not_said"}).status_code == 200
    assert post("Z9", {"action": "accept", "reason_code": "accurate"}).status_code == 404
    assert post("D1", {"action": "accept", "reason_code": ""}).status_code == 422

    minutes = store.latest_minutes(MID)
    assert minutes.review.status == "in_review"
    assert minutes.actions[0].description == "Draft the privacy notice"
    assert minutes.open_questions == []
    stored = store.list_review_items(MID, 1)
    assert [(i.item_id, i.action, i.reason_code) for i in stored] == [
        ("D1", "accept", "accurate"),
        ("A1", "modify", "wording"),
        ("Q1", "reject", "not_said"),
    ]
    assert stored[1].before == "Draft the notice" and stored[1].after == "Draft the privacy notice"
    assert stored[2].before == "Keep audio?"
    assert all(i.by == env["identity"].user and i.at == NOW for i in stored)
    assert [i.model_dump() for i in minutes.review.items] == [i.model_dump() for i in stored]
    audited = [e for e in events() if e["event"] == "review.item"]
    assert [(e["object"], e["detail"]["action"], e["detail"]["reason"]) for e in audited] == [
        ("D1", "accept", "accurate"),
        ("A1", "modify", "wording"),
        ("Q1", "reject", "not_said"),
    ]
    assert all(e["actor_source"] == "session" for e in audited)


class _BearerIdentity:
    """Service-mode stand-in: the identity comes from the request, never from a provider call
    without one (``OidcIdentity`` raises when there is no request)."""

    def current(self, request: Any = None) -> Any:
        if request is None:
            raise IdentityError("no request: bearer identity needs the Authorization header")
        token = request.headers.get("authorization", "")
        if token != "Bearer synthetic-ok":
            raise IdentityError("missing or bad bearer token")
        return FakeIdentity(
            user="r.haddad@acme.test",
            display="R. Haddad",
            source="oidc",
            groups=("Praktika-Secretaries",),
        ).current()


def test_audit_actor_is_the_request_identity_in_service_mode(
    tmp_settings: Settings, tmp_path: Path
) -> None:
    """With an ``AuditLog`` that has no provider of its own (as under OIDC), every server event
    is still recorded against the bearer identity of the request, not ``system``."""
    store = SqliteStore(tmp_path / "praktika.db")
    chain = tmp_settings.data_dir / "audit.jsonl"
    audit = AuditLog(JsonlAuditSink(chain), store, None)
    store.save_meeting(base_meeting().model_copy(update={"state": MeetingState.draft_ready}))
    store.save_transcript(make_transcript("en", n=4), delete_after=None)
    store.save_minutes(base_minutes())
    client = client_for(
        server.create_app(tmp_settings, store, _BearerIdentity(), audit, clock=lambda: NOW)
    )
    assert client.get(f"/api/meetings/{MID}").status_code == 401
    headers = {"Authorization": "Bearer synthetic-ok"}
    assert client.get(f"/api/meetings/{MID}", headers=headers).status_code == 200
    r = client.post(
        f"/api/minutes/{MID}/items/D1",
        headers=headers,
        json={"action": "accept", "reason_code": "accurate"},
    )
    assert r.status_code == 200
    lines = [json.loads(ln) for ln in chain.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["auth.denied", "review.opened", "review.item"]
    denied, *acted = lines
    assert (denied["actor"], denied["actor_source"]) == ("system", "system"), "nobody proven"
    assert denied["detail"]["reason"] == "unauthenticated" and denied["detail"]["status"] == 401
    assert denied["detail"]["route"] == f"GET /api/meetings/{MID}"
    by_reviewer = all(
        e["actor"] == "r.haddad@acme.test" and e["actor_source"] == "oidc" for e in acted
    )
    assert by_reviewer, lines
    assert store.list_review_items(MID, 1)[0].by == "r.haddad@acme.test"


def test_speaker_mapping_renames_transcript_and_refs(env: dict[str, Any]) -> None:
    client, store = env["client"], env["store"]
    segs = [
        s.model_copy(update={"speaker": "SPEAKER_00", "speaker_kind": "label"})
        if s.id == "S0001"
        else s
        for s in env["transcript"].segments
    ]
    store.save_transcript(
        env["transcript"].model_copy(update={"segments": segs}), delete_after=None
    )
    store.set_review_status(
        base_minutes(
            decisions=[
                Decision(
                    id="D1",
                    statement="Pilot approved",
                    kind="approved",
                    decided_by="Committee",
                    refs=[ref().model_copy(update={"speaker": "SPEAKER_00"})],
                )
            ]
        )
    )
    assert client.get(f"/api/meetings/{MID}").json()["unmapped_speakers"] == ["SPEAKER_00"]

    r = client.post(f"/api/meetings/{MID}/speakers", json={"SPEAKER_00": "L. Farouk", "X": " "})
    assert r.status_code == 200 and r.json()["mapped"] == 1
    seg = store.get_transcript(MID).by_id()["S0001"]
    assert (seg.speaker, seg.speaker_kind) == ("L. Farouk", "identity")
    assert store.latest_minutes(MID).decisions[0].refs[0].speaker == "L. Farouk"
    assert client.get(f"/api/meetings/{MID}").json()["unmapped_speakers"] == []
    assert any(
        e["event"] == "review.speaker_mapped" and e["detail"]["labels"] == ["SPEAKER_00"]
        for e in env["events"]()
    )


def test_regenerate_needs_client_and_bumps_version(env: dict[str, Any]) -> None:
    client, store = env["client"], env["store"]
    body = {"section": "decisions", "instruction": "Keep only decisions taken by the Chair."}
    assert client.post(f"/api/minutes/{MID}/regenerate", json=body).status_code == 503
    fake = FakeLLM()
    app = server.create_app(
        env["settings"], store, env["identity"], env["audit"], llm_client=fake, clock=lambda: NOW
    )
    with_llm = client_for(app)
    assert (
        with_llm.post(
            f"/api/minutes/{MID}/regenerate", json={"section": "agenda", "instruction": "x" * 5}
        ).status_code
        == 422
    )
    r = with_llm.post(f"/api/minutes/{MID}/regenerate", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["minutes"]["version"] == 2 and store.latest_minutes(MID).version == 2
    assert store.latest_minutes(MID).review.status == "draft"
    assert store.get_meeting(MID).state is MeetingState.draft_ready
    assert fake.calls and "Keep only decisions taken by the Chair." in fake.calls[0].user
    assert "Keep only decisions" not in fake.calls[0].system
    drafted = [e for e in env["events"]() if e["event"] == "minutes.drafted"]
    assert drafted and drafted[-1]["detail"]["version"] == 2
    calls = [e for e in env["events"]() if e["event"] == "llm.call"]
    assert calls and calls[-1]["actor"] == env["identity"].user, "regenerate is audited"
    assert calls[-1]["detail"]["schema"] == "ChunkFindings" and calls[-1]["prompt_sha"]
    assert store.search_index("pilot", include_private=False, limit=5) == [], "draft unindexed"


def test_discard_clears_search_index(env: dict[str, Any]) -> None:
    from praktika.llm.prompts import TEMPLATES
    from praktika.store import search

    client, store = env["client"], env["store"]
    approved = base_minutes().model_copy(
        update={"review": base_minutes().review.model_copy(update={"status": "approved"})}
    )
    assert search.index_minutes(store, approved, TEMPLATES[approved.meeting_type])
    assert store.search_index("pilot", include_private=False, limit=5)
    assert not search.indexable(base_minutes(), TEMPLATES[approved.meeting_type]), "drafts never"
    r = client.post(f"/api/minutes/{MID}/discard", json={"reason_code": "duplicate"})
    assert r.status_code == 200
    assert store.search_index("pilot", include_private=False, limit=5) == []


# --------------------------------------------------------------------------- audio


def test_audio_404_for_restricted_and_deleted(env: dict[str, Any], tmp_path: Path) -> None:
    client, store = env["client"], env["store"]
    url = f"/api/meetings/{MID}/audio"
    assert client.get(url, params={"start": 0, "end": 1}).status_code == 404  # never retained

    wav = tmp_path / "system.wav"
    _write_wav(wav, seconds=3.0)
    media_id = store.save_media(MID, wav, "ab" * 32, kind="audio", delete_after=None)
    r = client.get(url, params={"start": 0.5, "end": 1.5})
    assert r.status_code == 206 and r.headers["content-type"].startswith("audio/wav")
    data, rate = sf.read(io.BytesIO(r.content))
    assert rate == 16000 and len(data) == 16000
    assert client.get(url, params={"start": 2, "end": 1}).status_code == 422
    assert client.get(url, params={"start": 0, "end": 500}).status_code == 422
    r = client.get(url, params={"start": 2.5, "end": 4.5})  # clamped at the end of the file
    assert r.status_code == 206 and len(sf.read(io.BytesIO(r.content))[0]) == 8000

    store.mark_deleted("media", media_id, NOW)
    assert client.get(url, params={"start": 0, "end": 1}).status_code == 404

    restricted = base_meeting().model_copy(
        update={"id": "M-20260916-c3d4", "classification": Classification.restricted}
    )
    store.save_meeting(restricted)
    store.save_media(restricted.id, wav, "ab" * 32, kind="audio", delete_after=None)
    assert (
        client.get(
            f"/api/meetings/{restricted.id}/audio", params={"start": 0, "end": 1}
        ).status_code
        == 404
    )
    assert client.get(f"/api/meetings/{restricted.id}").json()["audio_available"] is False


# --------------------------------------------------------------------------- export


def test_export_draft_403_in_pilot(env: dict[str, Any], tmp_path: Path) -> None:
    client, events = env["client"], env["events"]
    assert env["settings"].pilot is True
    assert client.get(f"/api/export/{MID}.md").status_code == 403
    assert client.get(f"/api/export/{MID}.md", params={"allow_draft": "true"}).status_code == 403
    assert client.get(f"/api/export/{MID}.docx", params={"allow_draft": "true"}).status_code == 403
    assert not any(e["event"] == "export.written" for e in events())
    assert (
        not list((env["settings"].data_dir / "exports").glob("*"))
        if (env["settings"].data_dir / "exports").exists()
        else True
    )

    # Outside the pilot a query parameter alone is still refused: the deployment must opt in too.
    outside_no_opt_in = env["settings"].model_copy(update={"pilot": False})
    app_no_opt_in = server.create_app(
        outside_no_opt_in, env["store"], env["identity"], env["audit"], clock=lambda: NOW
    )
    assert (
        client_for(app_no_opt_in)
        .get(f"/api/export/{MID}.md", params={"allow_draft": "true"})
        .status_code
        == 403
    )

    outside = env["settings"].model_copy(update={"pilot": False, "allow_draft_export": True})
    app = server.create_app(outside, env["store"], env["identity"], env["audit"], clock=lambda: NOW)
    non_pilot = client_for(app)
    assert non_pilot.get(f"/api/export/{MID}.md").status_code == 403  # still needs allow_draft
    r = non_pilot.get(f"/api/export/{MID}.md", params={"allow_draft": "true"})
    assert r.status_code == 200 and "DRAFT" in r.text and "NOT APPROVED" in r.text
    written = [e for e in events() if e["event"] == "export.written"]
    assert len(written) == 1 and written[0]["detail"]["draft"] is True
    assert "watermark_draft" in written[0]["detail"]["obligations"]


def test_export_approved_has_provenance(env: dict[str, Any]) -> None:
    client, store, events = env["client"], env["store"], env["events"]
    assert _approve(client).status_code == 200
    minutes = store.latest_minutes(MID)

    r = client.get(f"/api/export/{MID}.md")
    assert r.status_code == 200
    text = r.text
    assert "## Provenance" in text and "DRAFT" not in text
    assert minutes.provenance.generator_model in text
    assert minutes.provenance.prompt_sha256 in text and minutes.provenance.transcript_sha256 in text
    assert f'filename="{MID}.v1.md"' in r.headers["content-disposition"]

    r = client.get(f"/api/export/{MID}.docx")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    path = env["settings"].data_dir / "exports" / f"{MID}.v1.docx"
    assert path.exists() and (path.stat().st_mode & 0o777) == 0o600

    written = [e for e in events() if e["event"] == "export.written"]
    assert [(e["detail"]["format"], e["detail"]["draft"]) for e in written] == [
        ("md", False),
        ("docx", False),
    ]
    assert all(e["classification"] == "internal" and e["meeting_id"] == MID for e in written)
    # approval indexed the minutes for cross-meeting search
    assert store.search_index("pilot", include_private=False, limit=5)


def test_static_index_served(env: dict[str, Any]) -> None:
    r = env["client"].get("/")
    assert r.status_code == 200 and "Praktika" in r.text and 'dir="auto"' in r.text
    assert env["client"].get("/static/app.js").status_code == 200
    assert env["client"].get("/static/style.css").status_code == 200
    assert env["client"].get("/api/me").json()["source"] == "session"


def test_wav_slice_helper_clamps(tmp_path: Path) -> None:
    wav = tmp_path / "t.wav"
    _write_wav(wav, seconds=1.0)
    assert len(sf.read(io.BytesIO(server.wav_slice(wav, 5.0, 6.0)))[0]) == 0
    assert len(sf.read(io.BytesIO(server.wav_slice(wav, 0.0, 0.25)))[0]) == 4000


def test_sorted_flags_and_item_kind() -> None:
    flags = [
        Flag(kind="possible_mnpi", detail="a", priority=3),
        Flag(
            kind="contradiction",
            detail="b",
            priority=2,
            cleared_by="x",
            cleared_at=NOW - timedelta(hours=1),
        ),
        Flag(kind="uncited_item_removed", detail="c", priority=1),
    ]
    rows = server.sorted_flags(base_minutes(flags=flags))
    assert [(r["n"], r["cleared"]) for r in rows] == [(2, False), (0, False), (1, True)]
    assert server.item_kind({"severity": "high", "description": "x"}) == "risks"
    assert server.item_kind({"description": "x", "owner_confidence": "explicit"}) == "actions"
    assert server.item_kind({"statement": "x"}) == "decisions"
    assert server.item_kind({"question": "x"}) == "open_questions"
    with pytest.raises(ValueError, match="no minutes list"):
        server.item_kind({"foo": 1})
    assert isinstance(make_transcript("en", n=1).segments[0], Segment)


def test_restore_assemble_stage_removal_and_reject_updates_commitments(
    env: dict[str, Any],
) -> None:
    """An item whose citations were all malformed (removed at assembly, not by the verifier)
    is stored as a complete item and can be restored; rejecting an action on a one-to-one
    removes it from the commitments too."""
    from praktika.llm import assemble
    from praktika.models import ActionDraft, DecisionDraft, MergedFindings, OneToOneMinutes

    client, store = env["client"], env["store"]
    by_id = env["transcript"].by_id()
    merged = MergedFindings(
        decisions=[
            DecisionDraft(
                statement="Malformed citation decision",
                kind="approved",
                decided_by="Chair",
                refs=["S12"],
                quote="whatever",
            )
        ],
        actions=[
            ActionDraft(
                description="Do the thing",
                owner="Omar Nasser",
                owner_confidence="explicit",
                refs=["S0002"],
                quote=by_id["S0002"].text,
            )
        ],
    )
    items, flags = assemble.assemble_items(merged, by_id, NOW.date())
    assert items["decisions"] == [] and len(flags) == 1
    from praktika.models import MeetingType

    one = OneToOneMinutes(
        **base_minutes(
            decisions=[],
            actions=items["actions"],
            flags=flags,
            meeting_type=MeetingType.one_to_one,
            version=2,
        ).model_dump(),
        my_commitments=[],
        their_commitments=items["actions"],
    )
    assert store.save_minutes(one) == 2
    detail = client.get(f"/api/meetings/{MID}").json()
    removed = [f for f in detail["flags"] if f["kind"] == "uncited_item_removed"][0]
    assert json.loads(removed["item_json"])["id"] == "D1"
    r = client.post(
        f"/api/minutes/{MID}/items/D1", json={"action": "restore", "reason_code": "cited_elsewhere"}
    )
    assert r.status_code == 200, r.text
    stored = store.latest_minutes(MID)
    assert [d.statement for d in stored.decisions] == ["Malformed citation decision"]
    assert stored.flags[0].cleared
    r = client.post(
        f"/api/minutes/{MID}/items/A1", json={"action": "reject", "reason_code": "not_said"}
    )
    assert r.status_code == 200, r.text
    stored = store.latest_minutes(MID)
    assert isinstance(stored, OneToOneMinutes)
    assert stored.actions == [] and stored.their_commitments == [] and stored.my_commitments == []
