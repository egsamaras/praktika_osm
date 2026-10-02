"""Shared builders for the core-module tests: meetings, settings, an in-memory retention store
and a tiny RS256 JWT signer (synthetic key generated per session; nothing real)."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conftest import FROZEN_NOW
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256

from praktika.audit import AuditLog, JsonlAuditSink, MemoryAuditStore
from praktika.config import Settings
from praktika.identity import FakeIdentity
from praktika.models import (
    Attendee,
    Classification,
    LanguageMode,
    Meeting,
    MeetingState,
    MeetingType,
    Platform,
)
from praktika.retention import AudioArtefact, Deletion, RetentionSubject

NOW = datetime.fromisoformat(FROZEN_NOW)
UTC_NOW = NOW.astimezone(UTC)


def make_settings(tmp_path: Path, **over: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "models_dir": tmp_path / "models",
        "llm_provider": "fake",
        "stt_en": "fake",
        "stt_ar": "fake",
        "identity_provider": "fake",
        "pilot_smoke": True,
    }
    base.update(over)
    return Settings(**base)


def make_meeting(**over: Any) -> Meeting:
    base: dict[str, Any] = {
        "id": "M-20260916-a1b2",
        "title": "Data team weekly",
        "meeting_type": MeetingType.general,
        "classification": Classification.internal,
        "language_mode": LanguageMode.ar_mixed,
        "platform": Platform.teams,
        "started_at": NOW,
        "organiser": "f.khalid@acme.test",
        "roster": [Attendee(name="F. Khalid", aliases=["فيصل"]), Attendee(name="R. Haddad")],
        "state": MeetingState.created,
    }
    base.update(over)
    return Meeting(**base)


def make_audit(tmp_path: Path, identity: FakeIdentity | None = None) -> tuple[AuditLog, Path]:
    path = tmp_path / "audit.jsonl"
    store = MemoryAuditStore()
    log = AuditLog(JsonlAuditSink(path), store, identity or FakeIdentity(source="session"))
    return log, path


class MemoryRetentionStore:
    """``RetentionStore`` over a list of subjects; ``record_deletion`` drops the artefact.

    ``on_record`` is called with the deletion before the row update so a test can check the
    file state at that instant.
    """

    def __init__(self, subjects: list[RetentionSubject]) -> None:
        self.subjects = {s.meeting.id: s for s in subjects}
        self.recorded: list[tuple[Deletion, datetime]] = []
        self.on_record: Any = None

    def retention_candidates(self, now: datetime) -> list[RetentionSubject]:
        return [s for s in self.subjects.values()]

    def record_deletion(self, deletion: Deletion, deleted_at: datetime) -> None:
        if self.on_record is not None:
            self.on_record(deletion)
        self.recorded.append((deletion, deleted_at))
        s = self.subjects[deletion.meeting_id]
        if deletion.kind == "audio":
            s.audio = [a for a in s.audio if a.path != deletion.path]
        elif deletion.kind == "transcript":
            s.transcript_created_at = None
        elif deletion.kind == "vault":
            s.vault_present = False
        elif deletion.kind == "draft":
            s.draft_created_at = None


def audio_file(tmp_path: Path, name: str, size: int = 4096) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x7f" * size)
    p.chmod(0o600)
    return p


def subject(meeting: Meeting, audio_paths: list[Path], created_at: datetime, **over: Any):
    return RetentionSubject(
        meeting=meeting,
        audio=[AudioArtefact(path=p, created_at=created_at) for p in audio_paths],
        **over,
    )


# --------------------------------------------------------------------------- JWT helpers


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _int_b64(n: int) -> str:
    return b64url(n.to_bytes((n.bit_length() + 7) // 8, "big"))


class Signer:
    """A synthetic RS256 signing key with a matching JWKS document."""

    def __init__(self, kid: str = "test-key-1") -> None:
        self.kid = kid
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwks(self) -> dict[str, Any]:
        pub = self.key.public_key().public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": self.kid,
                    "use": "sig",
                    "alg": "RS256",
                    "n": _int_b64(pub.n),
                    "e": _int_b64(pub.e),
                }
            ]
        }

    def token(self, claims: dict[str, Any], *, kid: str | None = None, alg: str = "RS256") -> str:
        header = {"alg": alg, "typ": "JWT", "kid": kid or self.kid}
        h = b64url(json.dumps(header, separators=(",", ":")).encode())
        p = b64url(json.dumps(claims, separators=(",", ":")).encode())
        sig = self.key.sign(f"{h}.{p}".encode("ascii"), padding.PKCS1v15(), SHA256())
        return f"{h}.{p}.{b64url(sig)}"


class Request:
    """Minimal stand-in for a Starlette request: only ``headers`` matters."""

    def __init__(self, **headers: str) -> None:
        self.headers = {k.replace("_", "-").lower(): v for k, v in headers.items()}
