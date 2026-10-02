"""Identifier tokenisation and the encrypted token vault (control C-06).

Contract: ``Tokeniser.apply`` replaces every detected identifier in a transcript with a stable
token such as ``«IBAN_1»`` before any LLM call, and returns the vault that maps tokens back to
plaintext. Tokenisation is idempotent (existing tokens are never rescanned) and roster names are
never tokenised. The vault is only ever persisted through ``encrypt_vault`` (Fernet). The
Keychain key is fetched lazily by ``keychain_key`` and never at import or construction time.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict, ValidationError

from praktika.errors import PraktikaError
from praktika.logging import get_logger
from praktika.models import Attendee, Transcript
from praktika.redact.names import name_near
from praktika.redact.normalise import arabic_indic_to_western
from praktika.redact.patterns import (
    KINDS,
    MASK,
    PATTERNS,
    TOKEN_PREFIX,
    iban_ok,
    luhn_ok,
    value_span,
)

log = get_logger(__name__)

TOKEN_RE = re.compile(r"«(?P<prefix>[A-Z]+)_(?P<n>\d+)»")
VAULT_KEY_ENV = "PRAKTIKA_VAULT_KEY"
_SECURITY = "/usr/bin/security"
_KEYCHAIN_ACCOUNT = "praktika"


class TokenVault(BaseModel):
    """Token -> plaintext map for one meeting. Plaintext holds identifiers; never log it."""

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    entries: dict[str, str] = {}


def _canonical(kind: str, value: str) -> str:
    """Key under which two spellings of the same identifier share one token."""
    if kind == "email":
        return f"{kind}:{value.lower()}"
    if kind == "amount_with_name":
        return f"{kind}:{re.sub(r'\s+', '', value).lower()}"
    return f"{kind}:{re.sub(r'[ -]', '', value).upper()}"


def _iban_end(candidate: str) -> int | None:
    """Length of the longest prefix of ``candidate`` (ending on a group) that is a valid IBAN."""
    ends = [m.end() for m in re.finditer(r"[A-Za-z0-9]+", candidate)]
    for end in reversed(ends):
        if iban_ok(candidate[:end]):
            return end
    return None


class Tokeniser:
    """Replace identifiers with ``«KIND_n»`` tokens, stable per value within one meeting.

    ``roster`` supplies the names, aliases and UPNs that must never be tokenised: roster words
    do not count as the "name" that turns an amount into ``«AMT_n»``, and roster UPNs are left
    as e-mail addresses because the roster is already visible to the model.
    """

    def __init__(self, roster: list[Attendee]) -> None:
        words: set[str] = set()
        for a in roster:
            for name in (a.name, *a.aliases):
                words.update(w.strip(".,") for w in name.split())
        self._known_tokens = frozenset(w for w in words if w)
        self._roster_emails = frozenset(a.upn.lower() for a in roster if a.upn)

    # ----------------------------------------------------------------- public API

    def apply(self, transcript: Transcript) -> tuple[Transcript, TokenVault]:
        """Return a redacted copy of ``transcript`` (``redacted=True``) and its vault.

        Segments are copied with ``model_copy`` so a token that is a few characters longer than
        the value it replaces cannot fail the 4000-character text bound; ids, times and speakers
        are untouched. Applying the result again is a no-op that yields an empty vault.
        """
        vault = TokenVault(meeting_id=transcript.meeting_id)
        segments = [
            s.model_copy(update={"text": self.apply_text(s.text, vault)})
            for s in transcript.segments
        ]
        counts: dict[str, int] = {}
        for token in vault.entries:
            prefix = TOKEN_RE.fullmatch(token).group("prefix")  # type: ignore[union-attr]
            counts[prefix] = counts.get(prefix, 0) + 1
        log.info("redact.applied", meeting_id=transcript.meeting_id, tokens=counts)
        return transcript.model_copy(update={"segments": segments, "redacted": True}), vault

    def apply_text(self, text: str, vault: TokenVault) -> str:
        """Tokenise one string, adding new tokens to ``vault`` and reusing existing ones.

        Kinds are applied in ``KINDS`` order and the whole pass repeats until nothing changes,
        so the result is a fixpoint: applying it again returns it unchanged. The loop always
        terminates because every pass either replaces plain text with a token (tokens are
        never rescanned) or leaves the text as it is.
        """
        reverse = {_canonical(_kind_of(t), v): t for t, v in vault.entries.items()}
        while True:
            before = text
            for kind in KINDS:
                text = self._apply_kind(text, kind, vault, reverse)
            if text == before:
                return text

    def detokenise(self, text: str, vault: TokenVault) -> str:
        """Replace every token present in ``vault`` with its plaintext; see ``detokenise``."""
        return detokenise(text, vault)

    # ----------------------------------------------------------------- internals

    def _apply_kind(self, text: str, kind: str, vault: TokenVault, reverse: dict[str, str]) -> str:
        """Scan ``text`` for one kind with existing tokens masked out, length preserved.

        Masking (rather than splitting at tokens) keeps context visible across a token: a name
        before an already-tokenised phone number still counts for an amount after it. Values
        never contain mask characters, so spliced spans always come from the original text.
        """
        if not text:
            return text
        masked = TOKEN_RE.sub(lambda m: MASK * len(m.group()), text)
        norm = arabic_indic_to_western(masked)
        out: list[str] = []
        pos = 0
        for m in PATTERNS[kind].finditer(norm):
            span = self._accept(kind, norm, m)
            if span is None or span[0] < pos:
                continue
            start, end = span
            value = norm[start:end]
            if MASK in value:
                continue
            token = self._token_for(kind, value, vault, reverse)
            out.append(text[pos:start])
            out.append(token)
            pos = end
        out.append(text[pos:])
        return "".join(out)

    def _accept(self, kind: str, norm: str, m: re.Match[str]) -> tuple[int, int] | None:
        start, end = value_span(m)
        value = norm[start:end]
        if kind == "iban":
            length = _iban_end(value)
            return None if length is None else (start, start + length)
        if kind == "card":
            return (start, end) if luhn_ok(value) else None
        if kind == "email":
            return None if value.lower() in self._roster_emails else (start, end)
        if kind == "amount_with_name":
            return (start, end) if name_near(norm, start, end, self._known_tokens) else None
        return (start, end)

    @staticmethod
    def _token_for(kind: str, value: str, vault: TokenVault, reverse: dict[str, str]) -> str:
        key = _canonical(kind, value)
        if key in reverse:
            return reverse[key]
        prefix = TOKEN_PREFIX[kind]
        used = [
            int(m.group("n"))
            for m in (TOKEN_RE.fullmatch(t) for t in vault.entries)
            if m and m.group("prefix") == prefix
        ]
        token = f"«{prefix}_{max(used, default=0) + 1}»"
        vault.entries[token] = value
        reverse[key] = token
        return token


def _kind_of(token: str) -> str:
    m = TOKEN_RE.fullmatch(token)
    if not m:
        raise PraktikaError(f"malformed vault token: {token!r}")
    for kind, prefix in TOKEN_PREFIX.items():
        if prefix == m.group("prefix"):
            return kind
    raise PraktikaError(f"unknown token kind in vault: {token!r}")


def detokenise(text: str, vault: TokenVault) -> str:
    """Return ``text`` with every token that ``vault`` knows replaced by its plaintext.

    Tokens absent from the vault are left as they are, so partial vaults never corrupt text.
    Callers must check ``policy.detokenise_allowed`` first; this function enforces nothing.
    """
    return TOKEN_RE.sub(lambda m: vault.entries.get(m.group(), m.group()), text)


def tokens_in(text: str) -> Iterable[str]:
    """Yield the tokens present in ``text`` in order of appearance."""
    return (m.group() for m in TOKEN_RE.finditer(text))


def encrypt_vault(vault: TokenVault, key: bytes) -> bytes:
    """Serialise ``vault`` as JSON and encrypt it with Fernet under ``key`` (a Fernet key).

    Raises ``PraktikaError`` when ``key`` is not a valid Fernet key.
    """
    return _fernet(key).encrypt(vault.model_dump_json().encode("utf-8"))


def decrypt_vault(blob: bytes, key: bytes) -> TokenVault:
    """Decrypt ``blob`` produced by ``encrypt_vault`` and validate it as a ``TokenVault``.

    Raises ``PraktikaError`` on a wrong key, a tampered blob or a payload that is not a vault.
    """
    try:
        payload = _fernet(key).decrypt(blob)
    except InvalidToken as e:
        raise PraktikaError("vault decryption failed: wrong key or tampered blob") from e
    try:
        return TokenVault.model_validate_json(payload)
    except ValidationError as e:
        raise PraktikaError("vault payload is not a TokenVault") from e


def _fernet(key: bytes) -> Fernet:
    try:
        return Fernet(key)
    except (ValueError, TypeError) as e:
        raise PraktikaError(
            "invalid vault key: expected a 32-byte urlsafe-base64 Fernet key"
        ) from e


def _security(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``/usr/bin/security``; a host without it (the Linux service image) raises
    ``PraktikaError`` naming ``PRAKTIKA_VAULT_KEY`` instead of a bare ``FileNotFoundError``."""
    try:
        # Fixed executable path and a fixed argument list; `service` is a caller-supplied label.
        return subprocess.run(args, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError as exc:
        raise PraktikaError(
            f"{VAULT_KEY_ENV} is not set and there is no key store on this host "
            f"({exc.strerror or exc}); set {VAULT_KEY_ENV} in the process environment"
        ) from exc


def keychain_key(service: str = "praktika-vault") -> bytes:
    """Return the vault key, creating one on first use. Lazy: never called at import time.

    Order: ``PRAKTIKA_VAULT_KEY`` from the process environment (service mode, injected by your
    secrets manager; validated, never persisted), else the macOS Keychain generic password
    ``service``/``praktika`` via ``/usr/bin/security``; when absent a fresh Fernet key is
    generated and stored there. Raises ``PraktikaError`` if the key is invalid or the Keychain is
    unavailable. Tests inject keys directly and never call this function.
    """
    env_key = os.environ.get(VAULT_KEY_ENV)
    if env_key:
        _fernet(env_key.encode("utf-8"))
        return env_key.encode("utf-8")
    find = [_SECURITY, "find-generic-password", "-a", _KEYCHAIN_ACCOUNT, "-s", service, "-w"]
    result = _security(find)
    if result.returncode == 0 and result.stdout.strip():
        key = result.stdout.strip().encode("utf-8")
        _fernet(key)
        return key
    key = Fernet.generate_key()
    add = [
        _SECURITY,
        "add-generic-password",
        "-a",
        _KEYCHAIN_ACCOUNT,
        "-s",
        service,
        "-w",
        key.decode(),
        "-U",
    ]
    created = _security(add)
    if created.returncode != 0:
        raise PraktikaError(
            f"could not store the vault key in the platform key store: {created.stderr.strip()}"
        )
    log.info("vault.key_created", service=service)
    return key
