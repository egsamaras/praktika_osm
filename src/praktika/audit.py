"""Hash-chained audit log (C-08): every event carries the previous event's hash.

Contract: ``AuditLog.append`` builds an ``AuditEvent`` whose ``prev_hash`` is the store's last
hash (``GENESIS`` for the first event), seals it, writes it to the store and then to the sink,
and returns it.

Concurrency. On the target host several processes append to one data directory at once (the
review server's worker threads, the retention timer, an ingest run). The chain step (read the
last hash, seal, store, emit) is therefore serialised at three levels: the log's thread lock;
the sink's cross-process lock (``JsonlAuditSink.lock``: an ``fcntl.flock`` on ``audit.jsonl.lock``
beside the log), held around the whole step so the JSONL file and the store's table receive
events in the same order; and the store's own write transaction (``SqliteStore.append_audit``
runs ``BEGIN IMMEDIATE``), which refuses a stale ``prev_hash`` so even a writer that bypasses
the file lock cannot fork the table's chain (``append`` re-reads the head and retries).

``verify_chain`` walks a JSONL file and reports the first line whose hash or chain link is
wrong; ``chain_report`` additionally cross-checks the file against the store's audit table and
the ``audit.jsonl.head`` file (under a shared lock, so it never reads a half-finished append),
so a truncated or deleted log is detected.

Sinks. ``audit.jsonl`` is always written and is always the authoritative record:
``JsonlAuditSink`` appends with ``fsync`` to a 0600 file and keeps the last hash in a 0600
``audit.jsonl.head`` beside it. ``ForwardingAuditSink`` does the same and also appends a copy of
each line to ``audit-forward.jsonl`` for the SIEM shipper; see its docstring for why the copy is
a dedicated file and never stdout or stderr. ``StdoutAuditSink`` is the name the
``audit_sink="stdout"`` setting has always selected; it now means "authoritative JSONL plus the
forwarded copy" (it no longer writes to stdout). ``audit_sink_for`` maps the setting to a sink.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from praktika.errors import PraktikaError
from praktika.identity import Identity, IdentityProvider
from praktika.logging import get_logger
from praktika.models import AuditEvent

log = get_logger(__name__)

GENESIS = "0" * 64
HEAD_SUFFIX = ".head"
LOCK_SUFFIX = ".lock"
AUDIT_FILE = "audit.jsonl"
FORWARD_FILE = "audit-forward.jsonl"
FORWARD_MAX_BYTES = 64 * 1024 * 1024
FORWARD_BACKUPS = 5
#: Attempts ``AuditLog.append`` makes when another writer moved the chain head under it.
CHAIN_RETRIES = 20
#: Longest wait for the audit file lock before the append (or report) gives up.
LOCK_TIMEOUT_S = 30.0
_LOCK_POLL_S = 0.01


class AuditLockTimeoutError(PraktikaError):
    """The audit file lock was not acquired within ``LOCK_TIMEOUT_S`` (a writer is hung)."""


class AuditSink(Protocol):
    def emit(self, e: AuditEvent) -> None: ...


@runtime_checkable
class LockingAuditSink(Protocol):
    """A sink that can hold a cross-process lock around the whole chain step."""

    def emit(self, e: AuditEvent) -> None: ...

    def lock(self) -> AbstractContextManager[None]: ...


class AuditStore(Protocol):
    """The slice of the store the audit log needs (implemented by ``SqliteStore``)."""

    def append_audit(self, event: AuditEvent) -> str: ...

    def last_audit_hash(self) -> str | None: ...


def _line(e: AuditEvent) -> str:
    return json.dumps(e.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def head_path(path: Path) -> Path:
    """The head file beside ``path`` that records the last hash written to it
    (``audit.jsonl.head`` for ``audit.jsonl``)."""
    return Path(path).with_name(Path(path).name + HEAD_SUFFIX)


def lock_path(path: Path) -> Path:
    """The ``audit.jsonl.lock`` file whose ``flock`` serialises writers of ``path``."""
    return Path(path).with_name(Path(path).name + LOCK_SUFFIX)


def _write_0600(path: Path, data: bytes, *, append: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _flock(path: Path, operation: int, timeout: float | None = None) -> Iterator[None]:
    """Hold ``fcntl.flock(operation)`` on ``path`` (created 0600) for the ``with`` block.

    ``flock`` locks belong to the open file description, so every acquisition opens its own
    descriptor: two threads or two sink objects in one process exclude each other exactly as two
    processes do. BSD ``flock`` on macOS and Linux ``flock`` on a local filesystem behave the
    same here; the data directory must not be on NFS. The lock is released when the descriptor
    closes, including when the process dies, so a crashed writer never wedges the log: the lock
    file stays behind, empty, and holds no lock. A *hung* holder is bounded instead: after
    ``timeout`` seconds (default ``LOCK_TIMEOUT_S``) ``AuditLockTimeoutError`` is raised, so a stuck
    writer or an accidental nested acquisition fails the audited action rather than blocking
    it forever.
    """
    limit = LOCK_TIMEOUT_S if timeout is None else timeout
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + limit
        while True:
            try:
                fcntl.flock(fd, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    log.error("audit.lock_timeout", path=str(path), waited_s=limit)
                    raise AuditLockTimeoutError(
                        f"audit log lock {path} not acquired within {limit:g} s"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class JsonlAuditSink:
    """Append one JSON line per event to ``path`` (created 0600), fsync after every write, and
    record the event's hash in ``audit.jsonl.head`` (0600, fsynced) so truncation of the log
    alone is detectable by ``chain_report``.

    ``lock()`` takes an exclusive ``flock`` on ``audit.jsonl.lock`` beside the log; ``AuditLog``
    holds it around the whole chain step and ``emit`` takes it too, so a direct ``emit`` is also
    safe. The lock is re-entrant within one thread of one sink object.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._held = threading.local()

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Exclusive cross-process lock on this log (re-entrant for the calling thread)."""
        depth: int = getattr(self._held, "depth", 0)
        if depth:
            self._held.depth = depth + 1
            try:
                yield
            finally:
                self._held.depth = depth
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _flock(lock_path(self.path), fcntl.LOCK_EX):
            self._held.depth = 1
            try:
                yield
            finally:
                self._held.depth = 0

    def emit(self, e: AuditEvent) -> None:
        with self.lock():
            self._write(e)

    def _write(self, e: AuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_0600(self.path, (_line(e) + "\n").encode("utf-8"), append=True)
        _write_0600(head_path(self.path), e.hash.encode("ascii"), append=False)


class ForwardingAuditSink(JsonlAuditSink):
    """``JsonlAuditSink`` plus a copy of every line in ``audit-forward.jsonl`` for the SIEM shipper.

    Why a dedicated file. The copy must never share a stream with anything else: stdout carries
    command output (``praktika config show`` prints JSON a script parses) and stderr carries the
    structured application log and ``error:`` messages, so either stream would interleave audit
    lines with other text, and the CLI and the retention timer have no process log for a shipper
    to tail at all. A file beside ``audit.jsonl`` works the same for every process (server, CLI,
    timer), is written under the same lock so it is in chain order, and is what Fluent Bit or
    the endpoint agent tails (``tail`` input with rotation handling).

    Contract. ``audit.jsonl`` and ``audit.jsonl.head`` are written first and stay authoritative:
    ``chain_report``, ``praktika audit verify`` and ``doctor`` read only them, so enabling the
    copy cannot make verification fail. The copy is best effort: an ``OSError`` while writing it
    is logged as ``audit.forward_failed`` (with the event hash, so the gap can be re-shipped from
    ``audit.jsonl``) and does not fail the audited action, which is already recorded. The copy is
    0600 and rotates by size: when the next line would take it past ``max_bytes`` it is renamed
    to ``.1`` (older copies shift up, at most ``backups`` are kept; ``backups=0`` discards).
    """

    def __init__(
        self,
        path: Path,
        forward_path: Path | None = None,
        *,
        max_bytes: int = FORWARD_MAX_BYTES,
        backups: int = FORWARD_BACKUPS,
    ) -> None:
        super().__init__(path)
        self.forward_path = (
            Path(forward_path) if forward_path else Path(path).with_name(FORWARD_FILE)
        )
        self.max_bytes, self.backups = max_bytes, backups

    def emit(self, e: AuditEvent) -> None:
        with self.lock():
            self._write(e)
            data = (_line(e) + "\n").encode("utf-8")
            try:
                self._rotate(len(data))
                _write_0600(self.forward_path, data, append=True)
            except OSError as exc:
                log.warning(
                    "audit.forward_failed", path=str(self.forward_path), hash=e.hash, error=str(exc)
                )

    def _rotate(self, incoming: int) -> None:
        try:
            size = self.forward_path.stat().st_size
        except FileNotFoundError:
            return
        if size == 0 or size + incoming <= self.max_bytes:
            return
        if self.backups <= 0:
            self.forward_path.unlink()
            return
        for i in range(self.backups - 1, 0, -1):
            older = self.forward_path.with_name(f"{self.forward_path.name}.{i}")
            if older.exists():
                older.replace(self.forward_path.with_name(f"{self.forward_path.name}.{i + 1}"))
        self.forward_path.replace(self.forward_path.with_name(f"{self.forward_path.name}.1"))
        log.info("audit.forward_rotated", path=str(self.forward_path))


class StdoutAuditSink(ForwardingAuditSink):
    """What ``audit_sink="stdout"`` selects: the authoritative JSONL plus the forwarded copy.

    The old meaning ("ship audit lines to the log collector") is kept; the mechanism changed.
    Nothing is written to stdout any more (it mixed audit JSON with command output) and
    ``audit.jsonl`` is always written (without it ``audit verify`` and ``doctor`` failed by
    construction). The collector tails ``audit-forward.jsonl`` instead of the process's stdout.

    ``StdoutAuditSink(path)`` is bound at once. ``StdoutAuditSink()`` (how the CLI has always
    built it) is bound by ``AuditLog`` to the store's data directory, i.e. the directory of the
    SQLite file, which is where ``audit.jsonl`` lives; with a store that has no file (``:memory:``,
    ``MemoryAuditStore``) ``AuditLog`` raises ``ValueError`` rather than write nowhere.
    """

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(Path(path) if path is not None else Path(AUDIT_FILE))
        self.bound = path is not None

    def bind_to_store(self, store: object) -> None:
        """Point an unbound sink at ``audit.jsonl`` beside ``store``'s database file."""
        if self.bound:
            return
        db = getattr(store, "path", None)
        if db is None or str(db) == ":memory:":
            raise ValueError(
                "audit_sink=stdout needs a data directory: pass StdoutAuditSink(path) "
                "or use a file-backed store"
            )
        self.path = Path(db).with_name(AUDIT_FILE)
        self.forward_path = self.path.with_name(FORWARD_FILE)
        self.bound = True

    @contextmanager
    def lock(self) -> Iterator[None]:
        """As ``JsonlAuditSink.lock``; refuses to run before the sink is bound."""
        if not self.bound:
            raise RuntimeError("StdoutAuditSink used before it was bound to a data directory")
        with super().lock():
            yield


def audit_sink_for(setting: str, data_dir: Path) -> JsonlAuditSink:
    """The sink for the ``audit_sink`` setting: ``jsonl`` writes ``audit.jsonl`` only;
    ``stdout`` (the historical name) and ``forward`` also write ``audit-forward.jsonl``."""
    path = Path(data_dir) / AUDIT_FILE
    if setting in ("stdout", "forward"):
        return ForwardingAuditSink(path)
    if setting == "jsonl":
        return JsonlAuditSink(path)
    raise ValueError(f"unknown audit_sink {setting!r}")


class MemoryAuditStore:
    """In-memory ``AuditStore`` for tests and for commands that run without a database."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append_audit(self, event: AuditEvent) -> str:
        self.events.append(event)
        return event.hash

    def last_audit_hash(self) -> str | None:
        return self.events[-1].hash if self.events else None

    def audit_count(self) -> int:
        return len(self.events)


class AuditLog:
    """Append hash-chained events on behalf of ``identity`` (or ``system`` when ``None``)."""

    def __init__(
        self,
        sink: AuditSink,
        store: AuditStore,
        identity: IdentityProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        bind = getattr(sink, "bind_to_store", None)
        if callable(bind):  # StdoutAuditSink() learns its data directory from the store
            bind(store)
        self._sink, self._store, self._identity = sink, store, identity
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()

    def _sink_lock(self) -> AbstractContextManager[None]:
        if isinstance(self._sink, LockingAuditSink):
            return self._sink.lock()
        return nullcontext()

    def _actor(self, actor: Identity | None) -> tuple[str, str]:
        if actor is not None:
            return actor.user, actor.audit_source()
        if self._identity is None:
            return "system", "system"
        who = self._identity.current()
        return who.user, who.audit_source()

    def append(
        self,
        event: str,
        meeting_id: str | None = None,
        *,
        actor: Identity | None = None,
        classification: str | None = None,
        object: str | None = None,  # noqa: A002 — field name fixed by AuditEvent
        model: str | None = None,
        prompt_sha: str | None = None,
        **detail: Any,
    ) -> AuditEvent:
        """Chain, seal, store and emit one event; ``detail`` must be JSON-serialisable.

        ``actor`` records the event on behalf of an identity already resolved by the caller (the
        review server resolves the bearer identity per request); otherwise the log's own
        provider is asked, and ``system`` is recorded when there is none.

        The chain step (read last hash, seal, store, emit) runs under the log's thread lock and
        the sink's cross-process lock, so concurrent appends from threads and from other
        processes produce one chained row each and the sink receives lines in chain order. If the
        store refuses the event because another writer moved the head in between (possible only
        for writers that do not share the sink's lock), the event is re-chained on the new head
        and retried, up to ``CHAIN_RETRIES`` times; any other store error propagates and nothing
        is emitted.
        """
        actor_name, source = self._actor(actor)
        with self._lock, self._sink_lock():
            for attempt in range(1, CHAIN_RETRIES + 1):
                prev = self._store.last_audit_hash() or GENESIS
                event_obj = AuditEvent(
                    ts=self._clock(),
                    actor=actor_name,
                    actor_source=source,  # type: ignore[arg-type]
                    event=event,
                    meeting_id=meeting_id,
                    classification=classification,
                    object=object,
                    detail=detail,
                    model=model,
                    prompt_sha=prompt_sha,
                    prev_hash=prev,
                ).sealed()
                try:
                    self._store.append_audit(event_obj)
                    break
                except ValueError:
                    moved = (self._store.last_audit_hash() or GENESIS) != prev
                    if not moved or attempt == CHAIN_RETRIES:
                        raise
                    log.info("audit.chain_retry", name=event, attempt=attempt)
            self._sink.emit(event_obj)
        log.debug("audit.appended", name=event, meeting_id=meeting_id)
        return event_obj

    @staticmethod
    def verify(path: Path) -> tuple[bool, int | None]:
        """Verify a JSONL chain file; see ``verify_chain``."""
        return verify_chain(path)


def _walk(path: Path) -> tuple[int | None, int, str]:
    """``(first bad line or None, events counted, last hash)`` for the JSONL chain at ``path``."""
    expected, count = GENESIS, 0
    with path.open(encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = AuditEvent.model_validate_json(line)
            except ValidationError:
                return n, count, expected
            if not event.verify_hash() or event.prev_hash != expected:
                return n, count, expected
            expected = event.hash
            count += 1
    return None, count, expected


def verify_chain(path: Path) -> tuple[bool, int | None]:
    """Return ``(True, None)`` if every line parses, self-verifies and chains from ``GENESIS``.

    Otherwise ``(False, n)`` where ``n`` is the 1-based number of the first bad line. Blank
    lines are ignored. A missing file counts as an empty, valid chain; use ``chain_report`` to
    detect a deleted or truncated log.
    """
    path = Path(path)
    if not path.exists():
        return True, None
    bad, _, _ = _walk(path)
    return (bad is None), bad


class AuditCountStore(Protocol):
    """A store that can also report how many audit rows it holds (``SqliteStore`` does)."""

    def last_audit_hash(self) -> str | None: ...

    def audit_count(self) -> int: ...


@dataclass(frozen=True)
class ChainReport:
    """Outcome of ``chain_report``: ``status`` is ``ok``, ``warn`` or ``fail``."""

    status: str
    detail: str
    bad_line: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def chain_report(path: Path, store: AuditCountStore | None = None) -> ChainReport:
    """Verify the JSONL chain and cross-check it against ``audit.jsonl.head`` and the store.

    A missing file is ``warn`` when nothing has been recorded yet and ``fail`` when the store
    or the head file say events exist. A file whose last hash differs from ``audit.jsonl.head`` or
    from ``store.last_audit_hash()``, or whose line count differs from ``store.audit_count()``,
    is reported as truncated (``fail``), because trailing lines cannot be missing otherwise.

    The file, the head and the store are read under a shared ``flock`` on the writers' lock
    file, so a report taken while other processes append sees a consistent snapshot instead of
    a false "truncated". A caller that cannot open the lock file (read-only copy of the data
    directory) or that times out waiting for a hung writer reads without it.
    """
    path = Path(path)
    with ExitStack() as stack:
        try:
            stack.enter_context(_flock(lock_path(path), fcntl.LOCK_SH))
        except (OSError, AuditLockTimeoutError) as exc:
            log.debug("audit.report_unlocked", path=str(path), error=str(exc))
        return _chain_report(path, store)


def _chain_report(path: Path, store: AuditCountStore | None) -> ChainReport:
    head = head_path(path)
    head_hash = head.read_text(encoding="utf-8").strip() if head.exists() else None
    stored_hash = store.last_audit_hash() if store is not None else None
    stored_count = store.audit_count() if store is not None else None
    if not path.exists():
        recorded = (head_hash is not None) or bool(stored_count)
        if recorded:
            return ChainReport("fail", f"{path} is missing but events were recorded")
        return ChainReport("warn", f"{path} does not exist yet (no events recorded)")
    bad, count, last = _walk(path)
    if bad is not None:
        return ChainReport("fail", f"chain broken at line {bad} of {path}", bad)
    if head_hash is not None and head_hash != last:
        return ChainReport("fail", f"{path} truncated: last hash differs from {head.name}")
    if stored_hash not in (None, GENESIS) and stored_hash != last:
        return ChainReport(
            "fail", f"{path} truncated at line {count} of {stored_count}: store hash differs"
        )
    if stored_count is not None and stored_count != count:
        return ChainReport(
            "fail", f"{path} holds {count} event(s) but the store holds {stored_count}"
        )
    return ChainReport("ok", f"{path} valid ({count} event(s))")
