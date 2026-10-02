"""Tests for ``praktika.redact`` (control C-06)."""

from __future__ import annotations

import re
import subprocess

import pytest
from conftest import make_transcript
from hypothesis import given, settings
from hypothesis import strategies as st

from praktika.errors import PraktikaError
from praktika.models import Attendee, Transcript
from praktika.redact import tokenise as tokenise_module
from praktika.redact.normalise import arabic_indic_to_western, normalise_number, strip_tashkeel
from praktika.redact.patterns import iban_ok, luhn_ok
from praktika.redact.tokenise import (
    TOKEN_RE,
    Tokeniser,
    TokenVault,
    decrypt_vault,
    detokenise,
    encrypt_vault,
    tokens_in,
)

MEETING = "M-20260916-a1b2"
# Every identifier in this file is fictional and cannot belong to anyone:
# - phones: +44 7700 900xxx is Ofcom's range reserved for drama; Bahrain's 30xx mobile block and
#   Saudi Arabia's 052 mobile prefix are not allocated to any operator (checked October 2026);
# - CPR numbers have the form YYMMNNNNC and these use birth month 13, which no CPR can carry;
# - the iqama and national ID numbers fail the Luhn check digit that real Saudi IDs pass;
# - the IBAN uses the test country code XX and the card is the well-known Visa test number.
IBAN = "XX55 NWND 0000 1234 5678 90"
BAD_IBAN = "XX56 NWND 0000 1234 5678 90"
CARD = "4111 1111 1111 1111"
BAD_CARD = "4111 1111 1111 1112"


@pytest.fixture
def tokeniser(roster: list[Attendee]) -> Tokeniser:
    return Tokeniser(roster)


def _apply(tokeniser: Tokeniser, text: str) -> tuple[str, TokenVault]:
    vault = TokenVault(meeting_id=MEETING)
    return tokeniser.apply_text(text, vault), vault


def _transcript(*texts: str) -> Transcript:
    t = make_transcript("en", n=len(texts), redacted=False)
    segments = [s.model_copy(update={"text": x}) for s, x in zip(t.segments, texts, strict=True)]
    return t.model_copy(update={"segments": segments})


# --------------------------------------------------------------------------- normalise


def test_arabic_indic_to_western_is_length_preserving() -> None:
    src = "الرقم ٨٥١٣١٢٣٤٥ و ۱۲۳ و ١٫٢ و ١٬٢٠٠"
    out = arabic_indic_to_western(src)
    assert out == "الرقم 851312345 و 123 و 1.2 و 1,200"
    assert len(out) == len(src)


def test_strip_tashkeel_and_normalise_number() -> None:
    assert strip_tashkeel("مُدَوِّن الـــمحضر") == "مدون المحضر"
    assert normalise_number("1,200,000.50") == "1200000.50"
    assert normalise_number("١٬٢٠٠٬٠٠٠") == "1200000"
    assert normalise_number("3,5") == "3.5"
    assert normalise_number(" -1 200 ") == "-1200"
    assert normalise_number("no digits") == "nodigits"


def test_validators() -> None:
    assert iban_ok(IBAN) and iban_ok(IBAN.replace(" ", "").lower())
    assert not iban_ok(BAD_IBAN)
    assert not iban_ok("XX55")  # too short
    assert not iban_ok("XX55 NWND 0000 1234 5678 90 1234 5678 9012 3456")  # too long
    assert luhn_ok(CARD) and luhn_ok("4111-1111-1111-1111")
    assert not luhn_ok(BAD_CARD)
    assert not luhn_ok("4111 1111 1111")  # 12 digits: below the PAN range
    assert not luhn_ok("41x1 1111 1111 1111")


# --------------------------------------------------------------------------- patterns


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            f"The IBAN is {IBAN}, thanks.", "The IBAN is «IBAN_1», thanks.", id="iban_mod97_valid"
        ),
        pytest.param(
            f"The IBAN is {BAD_IBAN}, thanks.",
            f"The IBAN is {BAD_IBAN}, thanks.",
            id="iban_mod97_invalid_not_tokenised",
        ),
        pytest.param(
            f"Card {CARD} expires; {BAD_CARD} is wrong.",
            f"Card «CARD_1» expires; {BAD_CARD} is wrong.",
            id="card_luhn",
        ),
        pytest.param(
            "CPR 851312345 and الرقم الشخصي 851312346 but 851312347 alone.",
            "CPR «CPR_1» and الرقم الشخصي «CPR_2» but 851312347 alone.",
            id="cpr_needs_context",
        ),
        pytest.param(
            "iqama 2123456789, هوية 1123456789, but iqama 3123456789.",
            "iqama «IQAMA_1», هوية «IQAMA_2», but iqama 3123456789.",
            id="iqama_prefix",
        ),
        pytest.param(
            "Call +973 3000 0123, +44 7700 900123, +966 52 123 4567, 0521234567 or 30000123.",
            "Call «PHONE_1», «PHONE_2», «PHONE_3», «PHONE_4» or «PHONE_5».",
            id="phone_forms",
        ),
        pytest.param(
            "Write to tom.brennan@northwind.test or tom at northwind.",
            "Write to «EMAIL_1» or tom at northwind.",
            id="email",
        ),
        pytest.param(
            "account number 12345678, a/c 12345678, رقم الحساب 987654321012, and 87654321 units.",
            "account number «ACC_1», a/c «ACC_1», رقم الحساب «ACC_2», and 87654321 units.",
            id="account_context_en_ar",
        ),
        pytest.param(
            f"transfer to {IBAN.replace(' ', '').lower()} today",
            "transfer to «IBAN_1» today",
            id="iban_lower_case",
        ),
        pytest.param(
            "رقمها الشخصي 881301234 ورقم إقامته ٢١٢٣٤٥٦٧٨٩ منتهي",
            "رقمها الشخصي «CPR_1» ورقم إقامته «IQAMA_1» منتهي",
            id="arabic_possessive_cpr_iqama",
        ),
        pytest.param(
            "رقم هويته 1098765432 موجود عندي والسي بي آر حقها 881301235",
            "رقم هويته «IQAMA_1» موجود عندي والسي بي آر حقها «CPR_1»",
            id="arabic_possessive_nid_and_spoken_cpr",
        ),
        pytest.param(
            "راتب فاطمة العلي 3,200 دينار شهرياً",
            "راتب فاطمة العلي «AMT_1» شهرياً",
            id="arabic_name_amount",
        ),
        pytest.param(
            "حوّل ٥٠٠٠ دينار للسيد أحمد يوسف عن فاتورة مارس",
            "حوّل «AMT_1» للسيد أحمد يوسف عن فاتورة مارس",
            id="arabic_honorific_amount",
        ),
        pytest.param(
            "الميزانية ٥٠٠ دينار بحريني للمرحلة الأولى",
            "الميزانية ٥٠٠ دينار بحريني للمرحلة الأولى",
            id="arabic_amount_without_name_untouched",
        ),
        pytest.param(
            "Pay BHD 5,000 to Ahmed Yusuf. The budget for the first phase of the racks is BHD 0.8m "
            "in total for the racks. Then 300 BHD for Hassan.",
            "Pay «AMT_1» to Ahmed Yusuf. The budget for the first phase of the racks is BHD 0.8m "
            "in total for the racks. Then «AMT_2» for Hassan.",
            id="amount_with_name",
        ),
        # The spoken English currency forms Whisper emits, and possessive names (C-06).
        pytest.param(
            "we pay Karim Mansour 4,500 dinars a month, and Hassan Ali earns BD 4,500 too",
            "we pay Karim Mansour «AMT_1» a month, and Hassan Ali earns «AMT_2» too",
            id="spoken_dinars_and_bd",
        ),
        pytest.param(
            "Layla Hassan owes 4500 riyals; Karim Mansour gets 4,500 Bahraini dinars",
            "Layla Hassan owes «AMT_1»; Karim Mansour gets «AMT_2»",
            id="riyals_and_qualified_dinars",
        ),
        pytest.param(
            "Karim's salary is BHD 4,500. Hassan’s bonus is SR 12,000 and 2 million dollars.",
            "Karim's salary is «AMT_1». Hassan’s bonus is «AMT_2» and «AMT_3».",
            id="possessive_name_opens_sentence",
        ),
        pytest.param(
            "the fee is 4,500 Bahraini dinars and the budget is 2 million dinars for the racks",
            "the fee is 4,500 Bahraini dinars and the budget is 2 million dinars for the racks",
            id="spoken_currency_without_name_untouched",
        ),
        pytest.param(
            "abd 500 for the SRT and Karim Mansour on the BDX 4500 bench",
            "abd 500 for the SRT and Karim Mansour on the BDX 4500 bench",
            id="currency_word_forms_are_bounded",
        ),
    ],
)
def test_pattern_positive_and_near_miss(tokeniser: Tokeniser, text: str, expected: str) -> None:
    out, vault = _apply(tokeniser, text)
    assert out == expected
    for token, value in vault.entries.items():
        assert TOKEN_RE.fullmatch(token)
        assert value in arabic_indic_to_western(text)
        assert value not in out


def test_arabic_indic_digits_normalised_first(tokeniser: Tokeniser) -> None:
    text = "الرقم الشخصي ٨٥١٣١٢٣٤٥ والهاتف ‎+٩٧٣ ٣٠٠٠ ٠١٢٣"
    out, vault = _apply(tokeniser, text)
    assert "«CPR_1»" in out and "«PHONE_1»" in out
    assert not re.search(r"[٠-٩]", out.replace("«", "").replace("»", ""))
    assert vault.entries["«CPR_1»"] == "851312345"
    assert vault.entries["«PHONE_1»"] == "+973 3000 0123"
    # the same identifier in Western digits shares the token
    again = tokeniser.apply_text("CPR 851312345", vault)
    assert again == "CPR «CPR_1»"


# --------------------------------------------------------------------------- tokeniser


def test_same_value_same_token(tokeniser: Tokeniser) -> None:
    t = _transcript(
        f"The IBAN is {IBAN}.",
        f"Again, {IBAN.replace(' ', '')} as I said, and card {CARD}.",
        f"A second IBAN: {'XX' + '55' + 'NWND00001234567890'} and mail Tom@Northwind.test.",
        "and tom@northwind.test once more",
    )
    out, vault = tokeniser.apply(t)
    texts = [s.text for s in out.segments]
    assert texts[0] == "The IBAN is «IBAN_1»."
    assert texts[1] == "Again, «IBAN_1» as I said, and card «CARD_1»."
    assert texts[2] == "A second IBAN: «IBAN_1» and mail «EMAIL_1»."
    assert texts[3] == "and «EMAIL_1» once more"
    assert set(vault.entries) == {"«IBAN_1»", "«CARD_1»", "«EMAIL_1»"}
    assert vault.entries["«IBAN_1»"] == IBAN  # first spelling seen is kept
    assert vault.meeting_id == MEETING


def test_apply_marks_redacted_and_keeps_structure(tokeniser: Tokeniser) -> None:
    t = _transcript("nothing sensitive here", "my line is +44 7700 900123")
    assert t.redacted is False
    out, vault = tokeniser.apply(t)
    assert out.redacted is True
    assert t.redacted is False, "input is not mutated"
    assert [s.id for s in out.segments] == [s.id for s in t.segments]
    assert [(s.start, s.end, s.speaker) for s in out.segments] == [
        (s.start, s.end, s.speaker) for s in t.segments
    ]
    assert out.segments[0].text == "nothing sensitive here"
    assert list(tokens_in(out.segments[1].text)) == ["«PHONE_1»"]
    assert out.sha256() != t.sha256()


def test_roster_names_never_tokenised(roster: list[Attendee], tokeniser: Tokeniser) -> None:
    names = [a.name for a in roster] + [al for a in roster for al in a.aliases]
    upns = [a.upn for a in roster if a.upn]
    text = (
        "Omar Nasser gets BHD 300 and Rania Haddad approves BHD 0.8m; ليلى فاروق takes 500 BHD. "
        + " ".join(names)
        + " "
        + " ".join(upns)
        + " but Ahmed Yusuf receives BHD 5,000 and vendor@northwind.test wrote."
    )
    out, vault = _apply(tokeniser, text)
    for name in names:
        assert name in out, name
    for upn in upns:
        assert upn in out, upn
    assert "BHD 300" in out and "BHD 0.8m" in out and "500 BHD" in out
    assert "«AMT_1»" in out and "«EMAIL_1»" in out
    assert set(vault.entries.values()) == {"BHD 5,000", "vendor@northwind.test"}
    for value in vault.entries.values():
        assert not any(n in value for n in names)


def test_round_trip_detokenise(tokeniser: Tokeniser) -> None:
    text = (
        f"IBAN {IBAN}, card {CARD}, CPR 851312345, iqama 2123456789, phone +973 3000 0123, "
        "mail tom@northwind.test, account number 12345678, and BHD 5,000 for Ahmed Yusuf."
    )
    out, vault = _apply(tokeniser, text)
    assert len(vault.entries) == 8
    assert not any(v in out for v in vault.entries.values())
    assert detokenise(out, vault) == text
    assert tokeniser.detokenise(out, vault) == text
    # tokens missing from a partial vault are left in place, never invented
    partial = TokenVault(meeting_id=MEETING, entries={"«IBAN_1»": IBAN})
    partly = detokenise(out, partial)
    assert IBAN in partly and "«CARD_1»" in partly
    # Arabic-Indic identifiers come back in Western digits (the normalised value is vaulted)
    out2, vault2 = _apply(tokeniser, "الرقم الشخصي ٨٥١٣١٢٣٤٥")
    assert detokenise(out2, vault2) == "الرقم الشخصي 851312345"


def test_name_window_spans_existing_tokens(tokeniser: Tokeniser) -> None:
    """A name on the far side of an already-placed token still qualifies an amount."""
    out, vault = _apply(tokeniser, "so Ahmed Yusuf called +973 3000 0123 about BHD 5,000 today")
    assert out == "so Ahmed Yusuf called «PHONE_1» about «AMT_1» today"
    # and a token counts as one word in the six-word window, exactly like the text it replaced
    out2, _ = _apply(tokeniser, "so Ahmed BHD 1 x x x x BHD 2")
    assert out2 == "so Ahmed «AMT_1» x x x x «AMT_2»"


def test_bad_token_in_vault_rejected() -> None:
    bad = TokenVault(meeting_id=MEETING, entries={"«NOPE_1»": "x"})
    with pytest.raises(PraktikaError, match="unknown token kind"):
        Tokeniser([]).apply_text("anything", bad)
    with pytest.raises(PraktikaError, match="malformed vault token"):
        Tokeniser([]).apply_text(
            "anything", TokenVault(meeting_id=MEETING, entries={"IBAN_1": "x"})
        )


# --------------------------------------------------------------------------- vault


def test_vault_encrypt_decrypt(fixed_key: bytes) -> None:
    vault = TokenVault(
        meeting_id=MEETING, entries={"«IBAN_1»": IBAN, "«PHONE_1»": "+973 3000 0123"}
    )
    blob = encrypt_vault(vault, fixed_key)
    assert isinstance(blob, bytes)
    assert IBAN.encode() not in blob and b"PHONE_1" not in blob and MEETING.encode() not in blob
    assert decrypt_vault(blob, fixed_key) == vault

    other_key = tokenise_module.Fernet.generate_key()
    with pytest.raises(PraktikaError, match="wrong key or tampered"):
        decrypt_vault(blob, other_key)
    tampered = blob[:-1] + (b"A" if blob[-1:] != b"A" else b"B")
    with pytest.raises(PraktikaError, match="wrong key or tampered"):
        decrypt_vault(tampered, fixed_key)
    with pytest.raises(PraktikaError, match="invalid vault key"):
        encrypt_vault(vault, b"not-a-fernet-key")
    with pytest.raises(PraktikaError, match="invalid vault key"):
        decrypt_vault(blob, b"short")
    # a valid ciphertext whose payload is not a vault is rejected too
    not_vault = tokenise_module.Fernet(fixed_key).encrypt(b'{"meeting_id": "x", "extra": 1}')
    with pytest.raises(PraktikaError, match="not a TokenVault"):
        decrypt_vault(not_vault, fixed_key)


def test_keychain_key_is_lazy(
    monkeypatch: pytest.MonkeyPatch, roster: list[Attendee], fixed_key: bytes
) -> None:
    """Nothing in the redaction path touches the Keychain: the key is injected by the caller."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("keychain access attempted during tokenisation")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(tokenise_module, "keychain_key", _forbidden)
    out, vault = Tokeniser(roster).apply(_transcript(f"IBAN {IBAN}"))
    assert decrypt_vault(encrypt_vault(vault, fixed_key), fixed_key) == vault
    assert out.redacted
    source = tokenise_module.__file__
    with open(source, encoding="utf-8") as fh:
        body = fh.read()
    assert body.count("keychain_key(") == 1, "keychain_key is defined once and never called here"


# --------------------------------------------------------------------------- idempotence


_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:+-/@()«»_\n٠١٢٣٤٥٦٧٨٩"
_WORDS = st.sampled_from(
    ["حساب", "الرقم الشخصي", "إقامة", "BHD", "SAR", "CPR", "iqama", "account", "a/c", "Ahmed"]
)
_KNOWN = st.sampled_from(
    [
        IBAN,
        CARD,
        BAD_CARD,
        "CPR 851312345",
        "الرقم الشخصي ٨٥١٣١٢٣٤٥",
        "iqama 2123456789",
        "+973 3000 0123",
        "+44 7700 900123",
        "0521234567",
        "tom@northwind.test",
        "account number 12345678",
        "BHD 5,000 to Ahmed Yusuf",
        "Ahmed BHD 1 x x x x BHD 2",
        "«IBAN_1»",
        "«CARD_9»",
        "Omar Nasser BHD 300",
    ]
)
_TEXT = st.lists(st.one_of(_KNOWN, _WORDS, st.text(_CHARS, max_size=12)), max_size=8).map(" ".join)


@settings(max_examples=200, deadline=None)
@given(_TEXT)
def test_tokenise_idempotent(roster: list[Attendee], text: str) -> None:
    tokeniser = Tokeniser(roster)
    v1 = TokenVault(meeting_id=MEETING)
    once = tokeniser.apply_text(text, v1)
    v2 = TokenVault(meeting_id=MEETING)
    twice = tokeniser.apply_text(once, v2)
    assert twice == once
    assert v2.entries == {}, "a redacted text yields nothing new"
    # and with the original vault, re-applying neither changes text nor grows the vault
    before = dict(v1.entries)
    assert tokeniser.apply_text(once, v1) == once
    assert v1.entries == before
    for token in tokens_in(once):
        assert token in v1.entries or token in text


def test_keychain_unavailable_is_a_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``security`` (the Linux image) and without PRAKTIKA_VAULT_KEY the error names
    the variable rather than crashing with a traceback after the meeting is stored."""
    monkeypatch.delenv(tokenise_module.VAULT_KEY_ENV, raising=False)
    monkeypatch.setattr(tokenise_module, "_SECURITY", "/nonexistent/security")
    with pytest.raises(PraktikaError, match="PRAKTIKA_VAULT_KEY"):
        tokenise_module.keychain_key()
    monkeypatch.setenv(tokenise_module.VAULT_KEY_ENV, "not-a-key")
    with pytest.raises(PraktikaError, match="invalid vault key"):
        tokenise_module.keychain_key()
