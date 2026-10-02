"""Glossary normalisation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import REPO, make_transcript
from pydantic import ValidationError

from praktika import glossary
from praktika.glossary import GlossaryEntry, apply, load, normalise_text

ENTRIES = [
    GlossaryEntry(canonical="Acme", misrenderings=["Akme", "Acmy", "A C M E"]),
    GlossaryEntry(canonical="ManCom", misrenderings=["Mancom", "Man Comm"]),
    GlossaryEntry(canonical="notetaker", misrenderings=["Notataka", "note tacker"]),
    GlossaryEntry(canonical="Kubernetes", misrenderings=["Cooper Netties"]),
]


def _seg(text: str, i: int = 1):
    seg = make_transcript("en", n=1).segments[0]
    return seg.model_copy(update={"id": f"S{i:04d}", "text": text})


def test_misrendering_replaced() -> None:
    segs = [
        _seg("We asked Akme and Mancom about the Notataka pilot."),
        _seg("The Cooper Netties cluster runs Man Comm papers, said the note tacker.", 2),
        _seg("Nothing to change here.", 3),
    ]
    out = apply(segs, ENTRIES)
    assert out[0].text == "We asked Acme and ManCom about the notetaker pilot."
    assert out[1].text == "The Kubernetes cluster runs ManCom papers, said the notetaker."
    assert out[2] is segs[2], "unchanged segments are returned as the same object"
    assert out[0].id == "S0001" and out[0].speaker == segs[0].speaker
    assert [s.text for s in segs] == [
        "We asked Akme and Mancom about the Notataka pilot.",
        "The Cooper Netties cluster runs Man Comm papers, said the note tacker.",
        "Nothing to change here.",
    ], "input segments are never mutated"


def test_fuzzy_match_respects_threshold_and_word_boundaries() -> None:
    # Close-but-not-equal spelling is caught at the default threshold...
    assert normalise_text("the Notatakas pilot", ENTRIES) == "the notetaker pilot"
    # ...but not at a stricter one.
    assert normalise_text("the Notatakas pilot", ENTRIES, threshold=100) == "the Notatakas pilot"
    # Substrings inside longer words are never replaced (partial_ratio alone would).
    assert normalise_text("Akmeville and Acmyton", ENTRIES) == "Akmeville and Acmyton"
    # Case matters: lower-case "akme" is not a listed misrendering.
    assert normalise_text("an akme here", ENTRIES) == "an akme here"


def test_tokens_untouched() -> None:
    text = "Pay «IBAN_1» via Akme, then «ACC_2» and «Akme_3» to Mancom."
    assert (
        normalise_text(text, ENTRIES)
        == "Pay «IBAN_1» via Acme, then «ACC_2» and «Akme_3» to ManCom."
    )
    trap = GlossaryEntry(canonical="X", misrenderings=["IBAN_1", "ACC"])
    assert normalise_text("«IBAN_1» «ACC_2»", [trap]) == "«IBAN_1» «ACC_2»"
    seg = _seg("«CARD_1» Akme")
    assert apply([seg], ENTRIES)[0].text == "«CARD_1» Acme"


def test_empty_inputs() -> None:
    assert apply([], ENTRIES) == []
    seg = _seg("Akme")
    assert apply([seg], [])[0] is seg
    assert normalise_text("", ENTRIES) == ""
    assert normalise_text("Akme", [GlossaryEntry(canonical="Acme", misrenderings=["  "])]) == "Akme"


def test_load_repo_glossary_and_sha() -> None:
    path = REPO / "glossary.yaml"
    entries, sha = load(path)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    by_name = {e.canonical: e for e in entries}
    assert "Acme Bank" in by_name and "ManCom" in by_name
    assert by_name["ManCom"].arabic_variants and "Mancom" in by_name["ManCom"].misrenderings


def test_sha_changes_with_file(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text("entries:\n  - canonical: Acme\n    misrenderings: [Akme]\n", encoding="utf-8")
    entries1, sha1 = load(path)
    assert entries1 == [GlossaryEntry(canonical="Acme", misrenderings=["Akme"])]
    assert load(path)[1] == sha1, "same bytes, same hash"
    path.write_text(
        "entries:\n  - canonical: Acme\n    misrenderings: [Akme, Acmy]\n", encoding="utf-8"
    )
    entries2, sha2 = load(path)
    assert sha2 != sha1 and entries2[0].misrenderings == ["Akme", "Acmy"]


def test_load_rejects_malformed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="entries"):
        load(bad)
    bad.write_text("entries:\n  - canonical: Acme\n    talkative: yes\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load(bad)
    bad.write_text("entries:\n  - canonical: ''\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load(bad)


def test_token_regex_matches_redaction_tokens() -> None:
    assert glossary.TOKEN_RE.findall("a «IBAN_1» b «ACC_12» c") == ["«IBAN_1»", "«ACC_12»"]


def test_ordinary_words_are_never_rewritten(tmp_path: Path) -> None:
    """A bare dictionary word cannot be a misrendering: the transcript is citation evidence."""
    entries, _ = load(REPO / "glossary.yaml")
    text = "Let us take it offline over dinner on Thursday."
    assert normalise_text(text, entries) == text
    assert normalise_text("the fee is five hundred Bahrain dinner", entries) == (
        "the fee is five hundred BHD"
    )
    assert normalise_text("we left the case file at the office near the board room", entries) == (
        "we left the case file at the office near the board room"
    )
    for entry in entries:
        for mis in entry.misrenderings:
            assert " " in mis or mis.lower() not in glossary.COMMON_WORDS, mis
    bad = tmp_path / "glossary.yaml"
    bad.write_text("entries:\n  - canonical: BHD\n    misrenderings: [dinner]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordinary word"):
        load(bad)
    ok = tmp_path / "ok.yaml"
    ok.write_text(
        "entries:\n  - canonical: BHD\n    misrenderings: ['Bahrain dinner']\n", encoding="utf-8"
    )
    assert load(ok)[0][0].misrenderings == ["Bahrain dinner"]
