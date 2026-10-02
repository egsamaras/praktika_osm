"""SQLite connection and schema migration.

Contract: ``connect`` returns a connection in WAL mode with foreign keys enforced, backed by a
file created with mode 0600 (the database holds transcripts and the encrypted vault, so it is
never group- or world-readable). ``migrate`` brings the schema to ``SCHEMA_VERSION`` idempotently:
a new database gets the whole current schema (``schema.sql``) in one step, and an existing one
gets each later version's statements in turn (version 2 adds the ``run_locks`` table), so a
database written by an earlier release keeps working.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praktika.logging import get_logger
from praktika.store.locks import RUN_LOCKS_SQL

SCHEMA_VERSION = 2
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
MEMORY = ":memory:"

log = get_logger(__name__)


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``+00:00`` offset."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def json_dumps(obj: Any) -> str:
    """Canonical JSON (sorted keys, UTF-8 preserved) for stored domain objects."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _create_private_file(path: Path) -> None:
    """Create ``path`` with mode 0600 if it does not exist; tighten the mode if it does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database at ``path`` and run migrations.

    WAL journal, ``foreign_keys=ON``, a 5 s busy timeout and ``sqlite3.Row`` rows. The database
    file (and therefore its ``-wal``/``-shm`` companions, which inherit its mode) is 0600.
    ``":memory:"`` is accepted for tests and skips the file-mode handling.

    The connection is opened with ``check_same_thread=False``: ``SqliteStore`` serialises every
    call under its own lock, so the review server's worker threads may use the store the CLI
    thread opened. Callers that use a raw connection across threads must lock it themselves.
    """
    target = str(path)
    if target != MEMORY:
        _create_private_file(Path(target))
    conn = sqlite3.connect(target, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if target != MEMORY:
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":  # pragma: no cover - platform dependent
            log.warning("store.wal_unavailable", journal_mode=mode)
    migrate(conn)
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    """Return the applied schema version, or 0 when the database is empty."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    got = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    return int(got or 0)


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database to ``SCHEMA_VERSION``; return the version.

    An empty database gets ``schema.sql``, which is the whole current schema, recorded as
    ``SCHEMA_VERSION``. A database at an earlier version gets each later version's statements
    from ``_upgrades`` in turn, each recorded as it is applied. Every statement is
    ``IF NOT EXISTS`` so applying one twice (two processes opening the database at once) is
    harmless.
    """
    have = current_version(conn)
    if have >= SCHEMA_VERSION:
        return have
    for version, sql in _migrations(have):
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, utcnow_iso()),
        )
        conn.commit()
        log.info("store.migrated", version=version)
    return SCHEMA_VERSION


def _upgrades() -> list[tuple[int, str]]:
    """The statements that bring a database from the version before each one to it."""
    return [(2, RUN_LOCKS_SQL)]


def _migrations(have: int) -> list[tuple[int, str]]:
    if have == 0:
        return [(SCHEMA_VERSION, SCHEMA_PATH.read_text(encoding="utf-8"))]
    return [(version, sql) for version, sql in _upgrades() if version > have]
