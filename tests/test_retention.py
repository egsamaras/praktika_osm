"""Retention timers, legal hold and deletion receipts (controls C-05, C-14)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from helpers_core import (
    MemoryRetentionStore,
    audio_file,
    make_audit,
    make_meeting,
    make_settings,
    subject,
)

from praktika import retention
from praktika.models import Classification, MeetingState
from praktika.retention import Deletion, RetentionPolicy, RetentionSubject

MID = {
    "internal": "M-20260916-0001",
    "confidential": "M-20260916-0002",
    "restricted": "M-20260916-0003",
}


@pytest.fixture
def policy(tmp_path: Path) -> RetentionPolicy:
    return RetentionPolicy.from_settings(make_settings(tmp_path))


def _now() -> datetime:
    return datetime.now(UTC)


def _kinds(plan: list[Deletion]) -> list[tuple[str, str]]:
    return [(d.meeting_id, d.kind) for d in plan]


def test_policy_from_settings(policy: RetentionPolicy) -> None:
    assert policy.audio_hours == {"internal": 24, "confidential": 72, "restricted": 0}
    assert policy.transcript_days["restricted"] == 7 and policy.draft_days["restricted"] == 14
    assert policy.transcript_max_days == {"internal": 60, "confidential": 60, "restricted": 30}


def test_audio_deleted_per_class(
    tmp_path: Path, frozen_clock: Any, policy: RetentionPolicy
) -> None:
    now = _now()
    subjects = [
        subject(
            make_meeting(id=MID[c], classification=Classification(c), state=MeetingState.in_review),
            [audio_file(tmp_path, f"{c}.wav")],
            created_at=now,
        )
        for c in ("internal", "confidential")
    ]
    store = MemoryRetentionStore(subjects)
    assert retention.plan(store, now, policy) == [], "fresh audio under review is kept"
    frozen_clock.tick(timedelta(hours=24))
    plan = retention.plan(store, _now(), policy)
    assert _kinds(plan) == [(MID["internal"], "audio")]
    assert "24 h" in plan[0].reason and plan[0].path == tmp_path / "internal.wav"
    frozen_clock.tick(timedelta(hours=48))
    assert _kinds(retention.plan(store, _now(), policy)) == [
        (MID["internal"], "audio"),
        (MID["confidential"], "audio"),
    ]


def test_audio_deleted_on_approval_or_discard_before_max(
    tmp_path: Path, policy: RetentionPolicy
) -> None:
    now = _now()
    approved = subject(
        make_meeting(id=MID["internal"], state=MeetingState.approved),
        [audio_file(tmp_path, "a.wav")],
        created_at=now,
    )
    discarded = subject(
        make_meeting(
            id=MID["confidential"],
            classification=Classification.confidential,
            state=MeetingState.discarded,
        ),
        [audio_file(tmp_path, "b.wav")],
        created_at=now,
    )
    plan = retention.plan(MemoryRetentionStore([approved, discarded]), now, policy)
    assert _kinds(plan) == [(MID["internal"], "audio"), (MID["confidential"], "audio")]
    assert plan[0].reason == "minutes approved" and plan[1].reason == "minutes discarded"


def test_restricted_audio_deleted_at_transcription(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()

    def restricted(state: MeetingState) -> RetentionSubject:
        m = make_meeting(
            id=MID["restricted"], classification=Classification.restricted, state=state
        )
        return subject(m, [audio_file(tmp_path, f"r_{state.value}.wav")], created_at=now)

    for state in (MeetingState.created, MeetingState.capturing, MeetingState.transcribing):
        assert retention.plan(MemoryRetentionStore([restricted(state)]), now, policy) == []
    for state in (MeetingState.drafting, MeetingState.draft_ready, MeetingState.in_review):
        plan = retention.plan(MemoryRetentionStore([restricted(state)]), now, policy)
        assert _kinds(plan) == [(MID["restricted"], "audio")] and "transcription" in plan[0].reason
    # Internal audio at the same instant and state is kept for review playback.
    internal = subject(
        make_meeting(id=MID["internal"], state=MeetingState.drafting),
        [audio_file(tmp_path, "i.wav")],
        created_at=now,
    )
    assert retention.plan(MemoryRetentionStore([internal]), now, policy) == []


def test_transcript_deleted_after_approval_window(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    approved_at = now - timedelta(days=13)

    def subj(cls: str, **over: Any) -> RetentionSubject:
        m = make_meeting(
            id=MID[cls], classification=Classification(cls), state=MeetingState.approved
        )
        base = {
            "transcript_created_at": now - timedelta(days=20),
            "approved_at": approved_at,
            "vault_present": True,
        }
        base.update(over)
        return subject(m, [], created_at=now, **base)

    assert retention.plan(MemoryRetentionStore([subj("internal")]), now, policy) == []
    plan = retention.plan(MemoryRetentionStore([subj("internal")]), now + timedelta(days=1), policy)
    assert _kinds(plan) == [(MID["internal"], "transcript"), (MID["internal"], "vault")]
    assert plan[0].reason == "approval + 14 days"
    # Restricted: approval + 7 days.
    plan = retention.plan(MemoryRetentionStore([subj("restricted")]), now, policy)
    assert _kinds(plan) == [(MID["restricted"], "transcript"), (MID["restricted"], "vault")]
    # Never approved: the hard maximum applies (60 days internal, 30 restricted).
    unapproved = subj("internal", approved_at=None, transcript_created_at=now - timedelta(days=59))
    unapproved.meeting = make_meeting(id=MID["internal"], state=MeetingState.in_review)
    assert retention.plan(MemoryRetentionStore([unapproved]), now, policy) == []
    plan = retention.plan(MemoryRetentionStore([unapproved]), now + timedelta(days=1), policy)
    assert [d.kind for d in plan] == ["transcript", "vault"] and "maximum" in plan[0].reason
    # No vault: only the transcript.
    plan = retention.plan(
        MemoryRetentionStore([subj("internal", vault_present=False)]),
        now + timedelta(days=1),
        policy,
    )
    assert [d.kind for d in plan] == ["transcript"]


def test_drafts_deleted_after_window(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    abandoned = subject(
        make_meeting(id=MID["internal"], state=MeetingState.draft_ready),
        [],
        created_at=now,
        draft_created_at=now - timedelta(days=30),
    )
    plan = retention.plan(MemoryRetentionStore([abandoned]), now, policy)
    assert (
        _kinds(plan) == [(MID["internal"], "draft")] and plan[0].reason == "draft_ready + 30 days"
    )
    recent = subject(
        make_meeting(id=MID["internal"], state=MeetingState.approved),
        [],
        created_at=now,
        draft_created_at=now - timedelta(days=45),
        approved_at=now - timedelta(days=29),
    )
    assert retention.plan(MemoryRetentionStore([recent]), now, policy) == []
    assert _kinds(
        retention.plan(MemoryRetentionStore([recent]), now + timedelta(days=1), policy)
    ) == [(MID["internal"], "draft")]


def test_legal_hold_blocks_all(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    long_ago = now - timedelta(days=400)
    held = subject(
        make_meeting(id=MID["internal"], state=MeetingState.approved, legal_hold=True),
        [audio_file(tmp_path, "held.wav")],
        created_at=long_ago,
        transcript_created_at=long_ago,
        vault_present=True,
        approved_at=long_ago,
        draft_created_at=long_ago,
    )
    free = subject(
        make_meeting(id=MID["confidential"], state=MeetingState.approved),
        [audio_file(tmp_path, "free.wav")],
        created_at=long_ago,
        transcript_created_at=long_ago,
        vault_present=True,
        approved_at=long_ago,
        draft_created_at=long_ago,
    )
    store = MemoryRetentionStore([held, free])
    plan = retention.plan(store, now, policy)
    assert {d.meeting_id for d in plan} == {MID["confidential"]}
    assert {d.kind for d in plan} == {"audio", "transcript", "vault", "draft"}
    audit, _ = make_audit(tmp_path)
    retention.run(store, audit, now, policy=policy)
    assert (tmp_path / "held.wav").exists() and not (tmp_path / "free.wav").exists()
    assert {d.meeting_id for d, _ in store.recorded} == {MID["confidential"]}


def test_idempotent_second_run(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    wav = audio_file(tmp_path, "old.wav")
    store = MemoryRetentionStore(
        [
            subject(
                make_meeting(id=MID["internal"], state=MeetingState.approved),
                [wav],
                created_at=now - timedelta(days=1),
                transcript_created_at=now - timedelta(days=30),
                approved_at=now - timedelta(days=20),
                vault_present=True,
            )
        ]
    )
    audit, path = make_audit(tmp_path)
    first = retention.run(store, audit, now, policy=policy)
    assert [d.kind for d in first] == ["audio", "transcript", "vault"]
    assert not wav.exists()
    assert retention.run(store, audit, now, policy=policy) == []
    assert len(store.recorded) == 3
    assert (
        sum(1 for line in path.read_text("utf-8").splitlines() if "retention.deleted" in line) == 3
    )


def test_run_tolerates_already_missing_file(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    gone = tmp_path / "gone.wav"
    store = MemoryRetentionStore(
        [
            subject(
                make_meeting(id=MID["internal"], state=MeetingState.discarded),
                [gone],
                created_at=now,
            )
        ]
    )
    audit, path = make_audit(tmp_path)
    done = retention.run(store, audit, now, policy=policy)
    assert [d.path for d in done] == [gone] and len(store.recorded) == 1
    assert json.loads(path.read_text("utf-8").splitlines()[-1])["detail"]["file_removed"] is False


def test_unlink_before_row_update(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    wav = audio_file(tmp_path, "cap.wav", size=3 * 1024 * 1024 + 17)
    store = MemoryRetentionStore(
        [
            subject(
                make_meeting(id=MID["internal"], state=MeetingState.approved), [wav], created_at=now
            )
        ]
    )
    seen: list[bool] = []
    store.on_record = lambda d: seen.append(wav.exists())
    audit, _ = make_audit(tmp_path)
    retention.run(store, audit, now, policy=policy)
    assert seen == [False], "the file must be gone before the store row is updated"


def test_wipe_overwrites_before_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = audio_file(tmp_path, "wipe.wav", size=5000)
    observed: list[bytes] = []
    real_unlink = Path.unlink

    def spy(self: Path, *a: Any, **kw: Any) -> None:
        observed.append(self.read_bytes())
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", spy)
    assert retention.wipe_file(wav) is True
    assert observed == [b"\0" * 5000] and not wav.exists()
    assert retention.wipe_file(wav) is False
    assert retention.wipe_file(tmp_path) is False


def test_receipts_audited(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    wav = audio_file(tmp_path, "r.wav")
    m = make_meeting(
        id=MID["confidential"],
        classification=Classification.confidential,
        state=MeetingState.approved,
    )
    store = MemoryRetentionStore(
        [
            subject(
                m,
                [wav],
                created_at=now,
                transcript_created_at=now - timedelta(days=30),
                approved_at=now - timedelta(days=15),
            )
        ]
    )
    audit, path = make_audit(tmp_path)
    retention.run(store, audit, now, policy=policy)
    events = [json.loads(x) for x in path.read_text("utf-8").splitlines()]
    assert [e["event"] for e in events] == ["retention.deleted", "retention.deleted"]
    audio_evt, transcript_evt = events
    assert audio_evt["meeting_id"] == MID["confidential"]
    assert audio_evt["classification"] == "confidential"
    assert audio_evt["object"] == str(wav) and audio_evt["detail"]["kind"] == "audio"
    assert audio_evt["detail"]["file_removed"] is True
    assert (
        transcript_evt["object"] == "transcript"
        and "approval" in transcript_evt["detail"]["reason"]
    )
    assert transcript_evt["prev_hash"] == audio_evt["hash"]


def test_dry_run_changes_nothing(tmp_path: Path, policy: RetentionPolicy) -> None:
    now = _now()
    wav = audio_file(tmp_path, "dry.wav")
    store = MemoryRetentionStore(
        [
            subject(
                make_meeting(id=MID["internal"], state=MeetingState.approved), [wav], created_at=now
            )
        ]
    )
    audit, path = make_audit(tmp_path)
    plan = retention.run(store, audit, now, dry_run=True, policy=policy)
    assert [d.kind for d in plan] == ["audio"]
    assert wav.exists() and store.recorded == [] and not path.exists()


def test_naive_datetimes_treated_as_utc(tmp_path: Path, policy: RetentionPolicy) -> None:
    naive_now = datetime(2026, 9, 17, 9, 0)
    s = subject(
        make_meeting(id=MID["internal"], state=MeetingState.in_review),
        [audio_file(tmp_path, "n.wav")],
        created_at=datetime(2026, 9, 16, 8, 0),
    )
    assert [d.kind for d in retention.plan(MemoryRetentionStore([s]), naive_now, policy)] == [
        "audio"
    ]


def test_stored_delete_after_caps_the_hard_maximum(tmp_path: Path, policy: RetentionPolicy) -> None:
    """The deadline written with the first transcript row wins when it is earlier than
    ``created_at + max_days`` (a later re-save must never extend it)."""
    now = _now()
    m = make_meeting(id=MID["internal"], state=MeetingState.in_review)
    s = subject(
        m,
        [],
        created_at=now,
        transcript_created_at=now - timedelta(days=2),  # the row a speaker mapping re-saved
        transcript_delete_after=now - timedelta(hours=1),  # the original 60-day deadline
        vault_present=True,
    )
    plan = retention.plan(MemoryRetentionStore([s]), now, policy)
    assert _kinds(plan) == [(MID["internal"], "transcript"), (MID["internal"], "vault")]
    assert "hard maximum 60 days" in plan[0].reason
    later = s.model_copy(update={"transcript_delete_after": now + timedelta(days=90)})
    assert retention.plan(MemoryRetentionStore([later]), now, policy) == [], "not yet due"
    plan = retention.plan(MemoryRetentionStore([later]), now + timedelta(days=59), policy)
    assert [d.kind for d in plan] == ["transcript", "vault"], "created_at + 60 still applies"


def test_failed_wipe_does_not_block_other_deletions(
    tmp_path: Path, policy: RetentionPolicy
) -> None:
    """A WAV that cannot be wiped is audited as ``retention.failed`` and left for the next
    run; every other deletion in the plan still happens and ``run`` then raises."""
    now = _now()
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked = audio_file(locked_dir, "first.wav")
    other = audio_file(tmp_path, "second.wav")
    subjects = [
        subject(
            make_meeting(id=MID["internal"], state=MeetingState.approved),
            [locked],
            created_at=now,
            transcript_created_at=now - timedelta(days=30),
            approved_at=now - timedelta(days=20),
        ),
        subject(
            make_meeting(
                id=MID["confidential"],
                classification=Classification.confidential,
                state=MeetingState.approved,
            ),
            [other],
            created_at=now,
        ),  # fmt: skip
    ]
    store = MemoryRetentionStore(subjects)
    audit, path = make_audit(tmp_path)
    locked.chmod(0o400)
    locked_dir.chmod(0o500)  # unlink would also fail: the directory is read-only
    try:
        with pytest.raises(retention.RetentionRunError, match="1 retention deletion") as info:
            retention.run(store, audit, now, policy=policy)
    finally:
        locked_dir.chmod(0o700)
        locked.chmod(0o600)
    err = info.value
    assert [(d.meeting_id, d.kind) for d, _ in err.failed] == [(MID["internal"], "audio")]
    assert [(d.meeting_id, d.kind) for d in err.done] == [
        (MID["internal"], "transcript"),
        (MID["confidential"], "audio"),
    ]
    assert not other.exists() and locked.exists(), "the other WAV was wiped; the locked one kept"
    assert [d.kind for d, _ in store.recorded] == ["transcript", "audio"]
    assert [a.path for a in store.subjects[MID["internal"]].audio] == [locked], "row stays live"
    events = [json.loads(x) for x in path.read_text("utf-8").splitlines()]
    assert [e["event"] for e in events] == [
        "retention.failed",
        "retention.deleted",
        "retention.deleted",
    ]
    failed = events[0]
    assert failed["object"] == str(locked) and failed["detail"]["kind"] == "audio"
    assert failed["detail"]["error"]
    # once the obstacle is gone the next run deletes it
    done = retention.run(store, audit, now, policy=policy)
    assert [(d.meeting_id, d.kind) for d in done] == [(MID["internal"], "audio")]
    assert not locked.exists()
