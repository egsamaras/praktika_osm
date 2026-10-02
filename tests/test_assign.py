"""Tests for ``praktika.diarize`` (control C-02)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from praktika.diarize import assign, base, pyannote_backend
from praktika.diarize.assign import apply_names, assign_speakers
from praktika.diarize.base import Turn
from praktika.errors import PraktikaError
from praktika.models import Segment, Word

DIARIZE_DIR = Path(__file__).resolve().parent.parent / "src" / "praktika" / "diarize"


def _seg(
    id_: str, start: float, end: float, text: str = "x", words: list[Word] | None = None
) -> Segment:
    return Segment(
        id=id_,
        start=start,
        end=end,
        speaker="SPEAKER_00",
        speaker_kind="label",
        language="en",
        text=text,
        track="file",
        words=words or [],
    )


def test_max_overlap() -> None:
    turns = [Turn(start=0.0, end=4.0, label="A"), Turn(start=4.0, end=10.0, label="B")]
    out = assign_speakers([_seg("S0001", 3.0, 8.0)], turns, split_on_words=False)
    assert [(s.speaker, s.speaker_kind) for s in out] == [("B", "label")]


def test_tie_break_earlier_turn() -> None:
    turns = [Turn(start=5.0, end=10.0, label="B"), Turn(start=0.0, end=5.0, label="A")]
    out = assign_speakers([_seg("S0001", 3.0, 7.0)], turns, split_on_words=False)
    assert out[0].speaker == "A", "equal overlap goes to the earlier turn regardless of input order"


def test_split_on_turn_boundary_with_words() -> None:
    words = [
        Word(start=0.2, end=1.0, text="We"),
        Word(start=1.0, end=2.0, text="agree."),
        Word(start=5.1, end=6.0, text="Thank"),
        Word(start=6.0, end=7.0, text="you."),
    ]
    turns = [Turn(start=0.0, end=5.0, label="A"), Turn(start=5.0, end=10.0, label="B")]
    before = [
        _seg("S0001", 0.0, 8.0, "We agree. Thank you.", words),
        _seg("S0002", 8.5, 9.5, "Bye"),
    ]

    out = assign_speakers(before, turns)

    assert [s.id for s in out] == ["S0001", "S0002", "S0003"], "renumbered after the split"
    assert [(s.speaker, s.text) for s in out] == [
        ("A", "We agree."),
        ("B", "Thank you."),
        ("B", "Bye"),
    ]
    assert (out[0].start, out[0].end) == (0.0, 2.0)
    assert (out[1].start, out[1].end) == (5.1, 8.0), "outer bounds of the segment are kept"
    assert [w.text for w in out[1].words] == ["Thank", "you."]
    assert out[0].language == "en" and out[0].track == "file"
    # Without splitting the same segment keeps one label (max overlap: A has 5 s, B has 3 s).
    whole = assign_speakers(before, turns, split_on_words=False)
    assert [s.id for s in whole] == ["S0001", "S0002"] and whole[0].speaker == "A"


def test_split_not_triggered_when_all_words_in_one_turn() -> None:
    words = [Word(start=1.0, end=2.0, text="a"), Word(start=2.0, end=3.0, text="b")]
    turns = [Turn(start=0.0, end=5.0, label="A"), Turn(start=5.0, end=9.0, label="B")]
    out = assign_speakers([_seg("S0007", 0.5, 3.5, "a b", words)], turns)
    assert len(out) == 1 and out[0].speaker == "A" and out[0].text == "a b" and out[0].id == "S0001"


def test_no_overlap_unknown() -> None:
    turns = [Turn(start=20.0, end=30.0, label="A")]
    out = assign_speakers([_seg("S0001", 0.0, 5.0)], turns)
    assert (out[0].speaker, out[0].speaker_kind) == ("unknown", "unknown")
    assert assign_speakers([_seg("S0001", 0.0, 5.0)], [])[0].speaker == "unknown"


def test_self_segments_untouched() -> None:
    mine = _seg("S0001", 0.0, 5.0).model_copy(update={"speaker": "ME", "speaker_kind": "self"})
    out = assign_speakers([mine], [Turn(start=0.0, end=5.0, label="A")])
    assert (out[0].speaker, out[0].speaker_kind) == ("ME", "self")


def test_apply_names_idempotent() -> None:
    segs = [
        _seg("S0001", 0.0, 1.0),
        _seg("S0002", 1.0, 2.0).model_copy(update={"speaker": "SPEAKER_01"}),
    ]
    mapping = {"SPEAKER_00": "F. Khalid", "SPEAKER_02": "R. Haddad", "SPEAKER_01": "  "}

    once = apply_names(segs, mapping)
    twice = apply_names(once, mapping)

    assert (once[0].speaker, once[0].speaker_kind) == ("F. Khalid", "identity")
    assert (once[1].speaker, once[1].speaker_kind) == ("SPEAKER_01", "label"), "blank name ignored"
    assert twice == once
    assert segs[0].speaker == "SPEAKER_00", "input not mutated"


def test_turn_model_has_no_embedding_field() -> None:
    assert set(Turn.model_fields) == {"start", "end", "label"}
    with pytest.raises(ValidationError):
        Turn(start=0.0, end=1.0, label="A", embedding=[0.1, 0.2])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Turn(start=2.0, end=1.0, label="A")
    for module in (base, assign, pyannote_backend):
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                assert not any("embed" in f for f in obj.model_fields), obj


def test_diarize_sources_never_touch_embeddings() -> None:
    for path in DIARIZE_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "speaker_embeddings" not in source and "embedding" not in source.lower(), path.name


def test_pyannote_backend_is_lazy_and_reports_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import types

    diarizer = pyannote_backend.PyannoteDiarizer(tmp_path, device="cpu")
    assert diarizer.name == "pyannote" and diarizer._pipeline is None
    diarizer.unload()  # never loaded: must not raise

    loaded: list[str] = []

    def stub_import(name: str, *a: object, **k: object) -> object:
        if name in ("pyannote.audio", "torch"):
            loaded.append(name)
            return types.SimpleNamespace(Pipeline=None, device=None)
        return importlib.import_module(name, *a, **k)

    monkeypatch.setattr("praktika.diarize.pyannote_backend.importlib.import_module", stub_import)
    with pytest.raises(PraktikaError, match="config not found"):
        diarizer.diarize(tmp_path / "x.wav")
    assert "pyannote.audio" in loaded, "weights are only touched on the first diarize()"
    assert diarizer._pipeline is None


def test_pyannote_backend_missing_extra_is_praktika_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    real = importlib.import_module

    def no_pyannote(name: str, *a: object, **k: object) -> object:
        if name.startswith("pyannote"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr("praktika.diarize.pyannote_backend.importlib.import_module", no_pyannote)
    with pytest.raises(PraktikaError, match="diarize"):
        pyannote_backend.PyannoteDiarizer(tmp_path).diarize(tmp_path / "x.wav")


def test_annotation_picker_prefers_exclusive() -> None:
    class Out:
        exclusive_speaker_diarization = "exclusive"
        speaker_diarization = "plain"

    class Plain:
        speaker_diarization = "plain"

    assert pyannote_backend._annotation(Out()) == "exclusive"
    assert pyannote_backend._annotation(Plain()) == "plain"
    with pytest.raises(PraktikaError):
        pyannote_backend._annotation(object())
