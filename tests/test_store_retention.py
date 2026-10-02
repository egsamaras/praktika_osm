"""Retention job and DSAR purge over the real ``SqliteStore`` (not the in-memory fake).

Proves the ``retention.RetentionStore`` seam end to end: subjects are built from stored rows,
``retention.run`` wipes files before the rows are retired, receipts are audited, a second run
deletes nothing, and ``purge_meeting`` erases every artefact while keeping the audit trail.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import make_transcript
from helpers_core import make_meeting, make_settings
from helpers_foundation import minutes as base_minutes

from praktika import retention
from praktika.audit import AuditLog, JsonlAuditSink
from praktika.errors import PraktikaError
from praktika.models import Classification, MeetingState, Review
from praktika.retention import RetentionPolicy
from praktika.store.repo import SqliteStore

MID = "M-20260916-a1b2"
NOW = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "praktika.db")
    yield s
    s.close()


@pytest.fixture
def policy(tmp_path: Path) -> RetentionPolicy:
    return RetentionPolicy.from_settings(make_settings(tmp_path))


def _audit(tmp_path: Path, store: SqliteStore) -> tuple[AuditLog, Path]:
    path = tmp_path / "audit.jsonl"
    return AuditLog(JsonlAuditSink(path), store, None), path


def _seed(store: SqliteStore, tmp_path: Path, **over: Any) -> Path:
    """A draft-ready internal meeting with one WAV, a transcript, a vault and a draft."""
    meeting = make_meeting(id=MID, state=MeetingState.draft_ready, **over)
    store.save_meeting(meeting)
    wav = tmp_path / "mic.wav"
    wav.write_bytes(b"\x7f" * 2048)
    store.save_media(MID, wav, "0" * 64, kind="mic", delete_after=None)
    store.save_transcript(make_transcript("en", n=3, meeting_id=MID), delete_after=None)
    store.save_vault(MID, b"blob")
    store.save_minutes(base_minutes())
    return wav


def test_subjects_built_from_rows(store: SqliteStore, tmp_path: Path) -> None:
    wav = _seed(store, tmp_path)
    store.save_meeting(make_meeting(id="M-20260916-0002"))  # nothing retained: not a subject
    subjects = store.retention_candidates(NOW)
    assert [s.meeting.id for s in subjects] == [MID]
    s = subjects[0]
    assert [a.path for a in s.audio] == [wav] and s.audio[0].created_at.tzinfo is not None
    assert s.transcript_created_at is not None and s.vault_present
    assert s.draft_created_at is not None and s.approved_at is None


def test_transcript_delete_after_follows_the_live_row(store: SqliteStore, tmp_path: Path) -> None:
    _seed(store, tmp_path)
    assert store.transcript_delete_after(MID) is None
    later = NOW + timedelta(days=14)
    store.save_transcript(make_transcript("en", n=2, meeting_id=MID), delete_after=later)
    assert store.transcript_delete_after(MID) == later
    assert store.transcript_delete_after("M-20260916-zzzz") is None


def test_approval_sets_approved_at_and_hides_draft_timer(
    store: SqliteStore, tmp_path: Path
) -> None:
    _seed(store, tmp_path)
    approved = base_minutes().model_copy(
        update={"review": Review(status="approved", reviewer="dpo", reviewed_at=NOW)}
    )
    store.set_review_status(approved)
    store.set_state(MID, MeetingState.approved)
    s = store.retention_candidates(NOW)[0]
    assert s.approved_at == NOW and s.draft_created_at is None


def test_run_over_sqlite_store_wipes_then_retires_rows(
    store: SqliteStore, tmp_path: Path, policy: RetentionPolicy
) -> None:
    wav = _seed(store, tmp_path)
    store.index_minutes(MID, 1, "Data team weekly", "summary", "body")
    audit, chain = _audit(tmp_path, store)
    assert retention.run(store, audit, NOW, policy=policy) == [], "fresh artefacts are kept"
    later = datetime.now(UTC) + timedelta(days=61)  # rows are stamped with the wall clock
    plan = retention.run(store, audit, later, dry_run=True, policy=policy)
    assert sorted(d.kind for d in plan) == ["audio", "draft", "transcript", "vault"]
    assert wav.exists() and store.get_transcript(MID) is not None, "dry run changes nothing"
    done = retention.run(store, audit, later, policy=policy)
    assert len(done) == 4 and not wav.exists()
    assert store.list_media(MID)[0].deleted_at is not None
    assert store.get_transcript(MID) is None and store.get_vault(MID) is None
    assert store.latest_minutes(MID) is None
    assert store.search_index("AI", include_private=True, limit=5) == [], "index row gone"
    row = store.conn.execute("SELECT segments_json FROM transcripts").fetchone()
    assert row["segments_json"] == "[]", "transcript content is erased, not only hidden"
    events = [json.loads(ln) for ln in chain.read_text("utf-8").splitlines()]
    receipts = [e for e in events if e["event"] == "retention.deleted"]
    assert len(receipts) == 4 and all(e["actor"] == "system" for e in receipts)
    assert {e["detail"]["kind"] for e in receipts} == {"audio", "draft", "transcript", "vault"}
    assert retention.run(store, audit, later, policy=policy) == [], "second run is idempotent"
    assert store.retention_candidates(later) == []


def test_restricted_audio_goes_at_transcription(
    store: SqliteStore, tmp_path: Path, policy: RetentionPolicy
) -> None:
    wav = _seed(store, tmp_path, classification=Classification.restricted)
    audit, _ = _audit(tmp_path, store)
    done = retention.run(store, audit, NOW, policy=policy)
    assert [d.kind for d in done] == ["audio"] and not wav.exists()
    assert store.get_transcript(MID) is not None


def test_legal_hold_keeps_everything(
    store: SqliteStore, tmp_path: Path, policy: RetentionPolicy
) -> None:
    wav = _seed(store, tmp_path)
    store.set_hold(MID, True, "litigation", "dpo")
    audit, _ = _audit(tmp_path, store)
    later = datetime.now(UTC) + timedelta(days=400)
    assert retention.run(store, audit, later, policy=policy) == []
    assert wav.exists() and store.get_transcript(MID) is not None
    store.set_hold(MID, False, "released", "dpo")
    assert len(retention.run(store, audit, later, policy=policy)) == 4


def test_purge_meeting_erases_artefacts_and_keeps_proof(store: SqliteStore, tmp_path: Path) -> None:
    wav = _seed(store, tmp_path)
    store.index_minutes(MID, 1, "Data team weekly", "summary", "body")
    assert store.purge_meeting(MID, NOW) == 1
    assert not wav.exists() and store.list_media(MID)[0].deleted_at is not None
    assert store.get_transcript(MID) is None and store.get_vault(MID) is None
    assert store.latest_minutes(MID) is None and store.list_review_items(MID, 1) == []
    assert store.search_index("summary", include_private=True, limit=5) == []
    assert store.get_meeting(MID).state is MeetingState.purged  # type: ignore[union-attr]
    assert store.get_meeting(MID) is not None, "the meeting row remains as proof"
    assert store.purge_meeting(MID, NOW) == 0, "purging twice is harmless"


def test_purge_refuses_hold_and_unknown(store: SqliteStore, tmp_path: Path) -> None:
    wav = _seed(store, tmp_path)
    store.set_hold(MID, True, "litigation", "dpo")
    with pytest.raises(PraktikaError, match="legal hold"):
        store.purge_meeting(MID, NOW)
    assert wav.exists()
    with pytest.raises(KeyError):
        store.purge_meeting("M-20260916-zzzz", NOW)


def test_store_is_usable_from_another_thread(store: SqliteStore, tmp_path: Path) -> None:
    """The review server's worker threads share the store the CLI thread opened."""
    import threading

    _seed(store, tmp_path)
    seen: list[Any] = []

    def worker() -> None:
        try:
            seen.append(store.get_meeting(MID))
        except Exception as exc:  # noqa: BLE001 - the test reports whatever escaped
            seen.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join(5)
    assert len(seen) == 1 and not isinstance(seen[0], Exception), seen
    assert seen[0].id == MID


def test_resaving_the_transcript_does_not_move_the_hard_maximum(
    store: SqliteStore, tmp_path: Path, policy: RetentionPolicy
) -> None:
    """A speaker mapping re-saves the transcript as a new row (with the original
    ``delete_after``); the 60-day hard maximum still counts from the first row."""
    from unittest.mock import patch

    _seed(store, tmp_path)
    first = NOW - timedelta(days=59)
    with patch("praktika.store.artefacts.utcnow_iso", return_value=first.isoformat()):
        store.save_transcript(
            make_transcript("en", n=3, meeting_id=MID), delete_after=first + timedelta(days=60)
        )
    # the fresh row of ``_seed`` (created "now") would restart the clock if it anchored the timer,
    # so retire it and keep only the dated one plus a later re-save
    store.conn.execute("DELETE FROM transcripts WHERE delete_after IS NULL")
    store.conn.commit()
    deadline = store.transcript_delete_after(MID)
    store.save_transcript(make_transcript("en", n=3, meeting_id=MID), delete_after=deadline)
    s = store.retention_candidates(NOW)[0]
    assert s.transcript_created_at == first and s.transcript_delete_after == deadline
    assert not [d for d in retention.plan(store, NOW, policy) if d.kind == "transcript"]
    due = retention.plan(store, NOW + timedelta(days=1, minutes=1), policy)
    kinds = {d.kind: d for d in due}
    assert {"transcript", "vault"} <= kinds.keys()
    assert "hard maximum 60 days" in kinds["transcript"].reason
