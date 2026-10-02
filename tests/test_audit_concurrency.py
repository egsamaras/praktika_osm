"""Cross-process audit appends (control C-08): several processes, one data directory.

The review server, the retention timer and an ingest run are separate processes that append to
the same ``praktika.db`` audit table and the same ``audit.jsonl``. These tests start real
processes (``spawn``, so macOS and Linux behave alike), release them together from a barrier and
check that no event is lost, the chain never forks and the JSONL file matches the table row for
row.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest

from praktika.audit import (
    GENESIS,
    AuditLog,
    JsonlAuditSink,
    chain_report,
    verify_chain,
)
from praktika.models import AuditEvent
from praktika.store.repo import SqliteStore

PROCESSES = 3
PER_PROCESS = 200
TIMEOUT_S = 120


class _NoLockSink:
    """A sink with no cross-process lock: only the database transaction protects the chain."""

    def emit(self, e: AuditEvent) -> None:
        return None


def _worker(data_dir: str, n: int, count: int, barrier: Barrier, with_file: bool) -> None:
    """Append ``count`` events as process ``n``; exit non-zero on any error."""
    data = Path(data_dir)
    store = SqliteStore(data / "praktika.db")
    sink = JsonlAuditSink(data / "audit.jsonl") if with_file else _NoLockSink()
    log = AuditLog(sink, store, None)
    barrier.wait(timeout=TIMEOUT_S)
    try:
        for i in range(count):
            log.append("review.item", "M-20260916-a1b2", proc=n, i=i)
    finally:
        store.close()


def _run(data: Path, *, with_file: bool) -> None:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(PROCESSES)
    procs = [
        ctx.Process(target=_worker, args=(str(data), n, PER_PROCESS, barrier, with_file))
        for n in range(PROCESSES)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(TIMEOUT_S)
    for p in procs:
        if p.is_alive():  # pragma: no cover - only on a hang
            p.kill()
    assert [p.exitcode for p in procs] == [0] * PROCESSES, "a writer process failed"


def _rows(data: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(data / "praktika.db")
    try:
        return [(r[0], r[1]) for r in conn.execute("SELECT prev_hash, hash FROM audit ORDER BY id")]
    finally:
        conn.close()


def _assert_table_chained(rows: list[tuple[str, str]]) -> None:
    total = PROCESSES * PER_PROCESS
    assert len(rows) == total, "every append from every process is stored exactly once"
    assert len({prev for prev, _ in rows}) == total, "no two rows share a prev_hash (no fork)"
    expected = GENESIS
    for prev, digest in rows:
        assert prev == expected
        expected = digest


@pytest.fixture
def data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(mode=0o700)
    SqliteStore(d / "praktika.db").close()  # migrate once before the writers start
    return d


def test_processes_share_one_chain_and_jsonl_matches_table(data: Path) -> None:
    _run(data, with_file=True)
    rows = _rows(data)
    _assert_table_chained(rows)
    path = data / "audit.jsonl"
    assert verify_chain(path) == (True, None)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [x["hash"] for x in lines] == [h for _, h in rows], "file and table in the same order"
    store = SqliteStore(data / "praktika.db")
    try:
        report = chain_report(path, store)
        assert report.ok, report.detail
    finally:
        store.close()


def test_database_transaction_alone_keeps_the_chain(data: Path) -> None:
    """Without the file lock, ``BEGIN IMMEDIATE`` in ``append_audit`` plus the log's retry still
    store every event once on one unforked chain."""
    _run(data, with_file=False)
    _assert_table_chained(_rows(data))
