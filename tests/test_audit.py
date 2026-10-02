"""Hash-chained audit log (control C-08)."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from praktika import audit as audit_mod
from praktika.audit import (
    GENESIS,
    AuditLog,
    JsonlAuditSink,
    MemoryAuditStore,
    StdoutAuditSink,
    chain_report,
    head_path,
    verify_chain,
)
from praktika.identity import FakeIdentity
from praktika.models import AuditEvent


def _log(tmp_path: Path, identity: Any = None) -> tuple[AuditLog, Path, MemoryAuditStore]:
    path = tmp_path / "audit.jsonl"
    store = MemoryAuditStore()
    return AuditLog(JsonlAuditSink(path), store, identity), path, store


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_chain_verifies(tmp_path: Path) -> None:
    log, path, store = _log(tmp_path, FakeIdentity(source="session"))
    e1 = log.append("consent.recorded", "M-20260916-a1b2", classification="internal", n=1)
    e2 = log.append("capture.started", "M-20260916-a1b2", object="mic.wav")
    e3 = log.append("models.verified", None, model="qwen2.5:14b", prompt_sha="a" * 64)
    assert e1.prev_hash == GENESIS
    assert e2.prev_hash == e1.hash and e3.prev_hash == e2.hash
    assert all(e.verify_hash() for e in (e1, e2, e3))
    assert store.last_audit_hash() == e3.hash
    assert AuditLog.verify(path) == (True, None)
    assert verify_chain(path) == (True, None)
    lines = _lines(path)
    assert [x["event"] for x in lines] == ["consent.recorded", "capture.started", "models.verified"]
    assert lines[2]["model"] == "qwen2.5:14b" and lines[2]["meeting_id"] is None


def test_tampered_line_detected(tmp_path: Path) -> None:
    log, path, _ = _log(tmp_path)
    for i in range(4):
        log.append("review.item", "M-20260916-a1b2", n=i)
    original = path.read_text(encoding="utf-8").splitlines()

    # 1. A field edited in the middle of the chain.
    lines = list(original)
    lines[1] = lines[1].replace('"n":1', '"n":99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditLog.verify(path) == (False, 2)

    # 2. A line removed: the next line's prev_hash no longer matches.
    lines = [original[0], *original[2:]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditLog.verify(path) == (False, 2)

    # 3. A line re-hashed after editing but not re-chained downstream.
    doc = json.loads(original[2])
    doc["detail"] = {"n": 42}
    doc["hash"] = AuditEvent(**doc).compute_hash()
    lines = [original[0], original[1], json.dumps(doc), original[3]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditLog.verify(path) == (False, 4)

    # 4. Garbage line.
    path.write_text("\n".join([original[0], "not json"]) + "\n", encoding="utf-8")
    assert AuditLog.verify(path) == (False, 2)

    # 5. First line not chained from genesis.
    path.write_text(original[1] + "\n", encoding="utf-8")
    assert AuditLog.verify(path) == (False, 1)


def test_verify_empty_and_missing_files(tmp_path: Path) -> None:
    assert verify_chain(tmp_path / "missing.jsonl") == (True, None)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    assert verify_chain(empty) == (True, None)


def test_event_schema_round_trip(tmp_path: Path) -> None:
    fixed = datetime(2026, 9, 16, 6, 0, tzinfo=UTC)
    log, path, _ = _log(tmp_path, FakeIdentity(source="session"))
    log._clock = lambda: fixed  # deterministic timestamp
    e = log.append(
        "llm.call",
        "M-20260916-a1b2",
        classification="internal",
        object="chunk-1",
        model="qwen2.5:14b",
        prompt_sha="b" * 64,
        tokens=1234,
        arabic="نعم",
    )
    line = path.read_text(encoding="utf-8").strip()
    back = AuditEvent.model_validate_json(line)
    assert back == e and back.verify_hash() and back.ts == fixed
    assert back.detail == {"tokens": 1234, "arabic": "نعم"}
    assert "\\u0646" not in line, "JSON lines keep UTF-8 (SIEM parsers expect it)"
    assert set(json.loads(line)) == set(AuditEvent.model_fields)


def test_actor_source_recorded(tmp_path: Path) -> None:
    log, _, store = _log(tmp_path, FakeIdentity(user="l.farouk@acme.test", source="session"))
    assert (log.append("x").actor, log.append("x").actor_source) == (
        "l.farouk@acme.test",
        "session",
    )
    log, _, _ = _log(tmp_path / "b", FakeIdentity(source="oidc"))
    assert log.append("auth.login").actor_source == "oidc"
    log, _, _ = _log(tmp_path / "c", FakeIdentity())
    assert log.append("x").actor_source == "local", "fake identity is recorded as local"
    log, _, _ = _log(tmp_path / "d", None)
    system = log.append("retention.deleted")
    assert (system.actor, system.actor_source) == ("system", "system")


def test_jsonl_sink_file_mode_0600_and_append(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"
    sink = JsonlAuditSink(path)
    log = AuditLog(sink, MemoryAuditStore(), None)
    log.append("a")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.chmod(0o644)
    log.append("b")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "mode re-tightened on every write"
    assert len(_lines(path)) == 2 and verify_chain(path) == (True, None)


def _gate_args() -> list[str]:
    return [
        "--notified",
        "--no-objections",
        "--method",
        "chat",
        "--teams-transcription-started",
        "--purpose",
        "Minutes for the data team weekly meeting",
        "--ack-all-scope",
    ]


def test_stdout_setting_keeps_command_output_clean_and_verifies(
    tmp_settings: Any, fixed_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``audit_sink=stdout`` (the SIEM setting) must not print audit JSON into command output,
    must still write the authoritative ``audit.jsonl`` so ``audit verify`` and ``doctor`` pass,
    and must put the forwarded copy in ``audit-forward.jsonl`` in chain order."""
    from conftest import FIXTURES, FakeLLM
    from typer.testing import CliRunner

    from praktika.cli import app, doctor
    from praktika.cli import context as ctx

    settings = tmp_settings.model_copy(update={"audit_sink": "stdout"})
    monkeypatch.setattr(ctx, "load_settings", lambda: settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    runner = CliRunner()
    args = [
        "ingest",
        str(FIXTURES / "synthetic_en.vtt"),
        "--title",
        "Data team weekly",
        "--roster",
        str(FIXTURES / "roster_data_team.yaml"),
        "--lang",
        "en",
        *_gate_args(),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert '"prev_hash"' not in result.stdout, "audit JSON leaked into command output"
    for line in result.stdout.splitlines():
        assert not line.lstrip().startswith('{"ts"'), line

    path = settings.data_dir / "audit.jsonl"
    forward = settings.data_dir / "audit-forward.jsonl"
    events = _lines(path)
    assert events and verify_chain(path) == (True, None)
    assert _lines(forward) == events, "the forwarded copy mirrors the chain, line for line"
    assert stat.S_IMODE(forward.stat().st_mode) == 0o600

    verify = runner.invoke(app, ["audit", "verify"])
    assert verify.exit_code == 0, verify.output
    assert '"prev_hash"' not in verify.stdout
    assert doctor.check_audit(settings).status == "ok"


def test_forwarding_sink_copy_matches_and_rotates(tmp_path: Path) -> None:
    from praktika.audit import ForwardingAuditSink

    path = tmp_path / "audit.jsonl"
    sink = ForwardingAuditSink(path, max_bytes=1500, backups=2)
    log = AuditLog(sink, MemoryAuditStore(), None)
    for i in range(12):
        log.append("export.written", "M-20260916-a1b2", n=i)
    assert verify_chain(path) == (True, None) and len(_lines(path)) == 12
    forward = tmp_path / "audit-forward.jsonl"
    rotated = [tmp_path / "audit-forward.jsonl.1", tmp_path / "audit-forward.jsonl.2"]
    assert forward.exists() and all(r.exists() for r in rotated)
    assert not (tmp_path / "audit-forward.jsonl.3").exists(), "only ``backups`` copies are kept"
    assert all(f.stat().st_size <= 1500 for f in [forward, *rotated])
    kept = _lines(rotated[1]) + _lines(rotated[0]) + _lines(forward)
    assert kept == _lines(path)[-len(kept) :], "rotation keeps the newest lines in order"
    assert chain_report(path, None).ok


def test_forward_copy_failure_does_not_break_the_chain(tmp_path: Path) -> None:
    from praktika.audit import ForwardingAuditSink

    path = tmp_path / "audit.jsonl"
    blocked = tmp_path / "blocked"
    blocked.mkdir()  # a directory cannot be appended to: every forward write fails
    log = AuditLog(ForwardingAuditSink(path, blocked), MemoryAuditStore(), None)
    e = log.append("export.written", "M-20260916-a1b2")
    assert _lines(path)[-1]["hash"] == e.hash and verify_chain(path) == (True, None)


def test_stdout_sink_binds_to_the_store_directory(tmp_path: Path) -> None:
    from praktika.store.repo import SqliteStore

    store = SqliteStore(tmp_path / "praktika.db")
    try:
        log = AuditLog(StdoutAuditSink(), store, None)
        e = log.append("export.written", "M-20260916-a1b2")
    finally:
        store.close()
    assert _lines(tmp_path / "audit.jsonl")[-1]["hash"] == e.hash
    assert _lines(tmp_path / "audit-forward.jsonl")[-1]["hash"] == e.hash
    with pytest.raises(ValueError, match="data directory"):
        AuditLog(StdoutAuditSink(), MemoryAuditStore(), None)


def test_audit_sink_for_maps_the_setting(tmp_path: Path) -> None:
    from praktika.audit import ForwardingAuditSink, audit_sink_for

    jsonl = audit_sink_for("jsonl", tmp_path)
    assert type(jsonl) is JsonlAuditSink and jsonl.path == tmp_path / "audit.jsonl"
    for name in ("stdout", "forward"):
        sink = audit_sink_for(name, tmp_path)
        assert isinstance(sink, ForwardingAuditSink)
        assert sink.forward_path == tmp_path / "audit-forward.jsonl"
    with pytest.raises(ValueError):
        audit_sink_for("syslog", tmp_path)


def test_sink_lock_excludes_a_second_sink_object(tmp_path: Path) -> None:
    """The file lock is per open file description, so a second sink object (a second
    ``AuditLog`` in the same process, as ``sweep_retention`` builds) waits for the first."""
    import threading

    path = tmp_path / "audit.jsonl"
    first, second = JsonlAuditSink(path), JsonlAuditSink(path)
    acquired = threading.Event()

    def contender() -> None:
        with second.lock():
            acquired.set()

    with first.lock():
        with first.lock():  # re-entrant for the holding thread
            t = threading.Thread(target=contender)
            t.start()
            assert not acquired.wait(0.3), "second sink got the lock while the first held it"
    t.join(5)
    assert acquired.is_set()


def test_hung_lock_holder_fails_the_append_instead_of_blocking_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the flock wait was unbounded, so a hung writer (or a nested acquisition
    through a second sink object in one thread) blocked every audited action for ever. Now the
    append fails with ``AuditLockTimeoutError`` and nothing is stored or written."""
    import time

    monkeypatch.setattr(audit_mod, "LOCK_TIMEOUT_S", 0.2)
    log, path, store = _log(tmp_path)
    holder = JsonlAuditSink(path)
    with holder.lock():
        started = time.monotonic()
        with pytest.raises(audit_mod.AuditLockTimeoutError):
            log.append("review.item", "M-20260916-a1b2")
        assert time.monotonic() - started < 5
    assert store.audit_count() == 0 and not path.exists()
    log.append("review.item", "M-20260916-a1b2")  # the lock is free again once released
    assert verify_chain(path) == (True, None)


def test_chain_continues_from_store_last_hash(tmp_path: Path) -> None:
    store = MemoryAuditStore()
    first = AuditLog(JsonlAuditSink(tmp_path / "a.jsonl"), store, None)
    e1 = first.append("a")
    second = AuditLog(JsonlAuditSink(tmp_path / "a.jsonl"), store, None)
    e2 = second.append("b")
    assert e2.prev_hash == e1.hash and verify_chain(tmp_path / "a.jsonl") == (True, None)


def test_concurrent_appends_stay_chained(tmp_path: Path) -> None:
    """N threads appending through one ``AuditLog`` over a real ``SqliteStore`` produce N
    chained rows and a JSONL file in chain order (no 'audit chain broken', no reordering)."""
    import threading

    from praktika.store.repo import SqliteStore

    store = SqliteStore(tmp_path / "praktika.db")
    path = tmp_path / "audit.jsonl"
    log = AuditLog(JsonlAuditSink(path), store, None)
    errors: list[BaseException] = []
    per_thread, threads = 50, 4

    def worker(n: int) -> None:
        try:
            for i in range(per_thread):
                log.append("review.item", "M-20260916-a1b2", thread=n, i=i)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert errors == []
    assert store.audit_count() == per_thread * threads
    assert verify_chain(path) == (True, None)
    lines = _lines(path)
    assert len(lines) == per_thread * threads
    assert lines[-1]["hash"] == store.last_audit_hash()
    assert chain_report(path, store).ok


def test_chain_report_detects_truncation_and_deletion(tmp_path: Path) -> None:
    log, path, store = _log(tmp_path)
    assert chain_report(path, store).status == "warn", "nothing recorded yet is a warning"
    for i in range(5):
        log.append("export.written", "M-20260916-a1b2", n=i)
    assert chain_report(path, store).ok
    assert stat.S_IMODE(head_path(path).stat().st_mode) == 0o600
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
    assert verify_chain(path) == (True, None), "a prefix still chains from genesis"
    report = chain_report(path, store)
    assert report.status == "fail" and "truncated" in report.detail
    # without the store, the head file alone reveals the truncation
    report = chain_report(path, None)
    assert report.status == "fail" and "truncated" in report.detail
    path.unlink()
    report = chain_report(path, store)
    assert report.status == "fail" and "missing" in report.detail
    assert chain_report(path, None).status == "fail"
