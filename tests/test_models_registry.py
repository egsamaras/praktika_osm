"""Model register: pull, register, verify."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from praktika.config import Settings
from praktika.errors import ModelRegisterMismatch
from praktika.models_registry import manage

TOKEN = "hf_synthetic_token_value_0123456789"  # noqa: S105 - fake value for the leak test


def fake_weights(
    root: Path, names: tuple[str, ...] = ("config.json", "weights.safetensors")
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        (root / name).write_bytes(f"synthetic-{name}-{i}".encode() * 100)
    (root / ".cache").mkdir(exist_ok=True)
    (root / ".cache" / "lock").write_text("hub bookkeeping")


def test_register_and_verify_round_trip(tmp_settings: Settings) -> None:
    root = tmp_settings.models_dir / "stt_en"
    fake_weights(root)
    rec = manage.register("stt_en", root, tmp_settings)
    assert rec.repo == "mlx-community/whisper-large-v3-turbo" and rec.licence == "MIT"
    assert sorted(rec.files_sha256) == ["config.json", "weights.safetensors"]  # hidden skipped
    assert manage.register_path(tmp_settings).exists()
    assert [m.role for m in manage.verify(tmp_settings)] == ["stt_en"]
    loaded = manage.load_register(tmp_settings).by_role()["stt_en"]
    assert loaded.files_sha256 == rec.files_sha256 and loaded.local_path == root.resolve()


def test_verify_detects_hash_mismatch(tmp_settings: Settings) -> None:
    root = tmp_settings.models_dir / "stt_ar"
    fake_weights(root)
    manage.register("stt_ar", root, tmp_settings, revision="abc123")
    (root / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelRegisterMismatch, match="hash mismatch: weights.safetensors"):
        manage.verify(tmp_settings)
    manage.register("stt_ar", root, tmp_settings)
    (root / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(ModelRegisterMismatch, match="unregistered files"):
        manage.verify(tmp_settings)
    (root / "extra.bin").unlink()
    (root / "config.json").unlink()
    with pytest.raises(ModelRegisterMismatch, match="file missing: config.json"):
        manage.verify(tmp_settings)
    manage.register_path(tmp_settings).unlink()
    with pytest.raises(ModelRegisterMismatch, match="no models registered"):
        manage.verify(tmp_settings)


def test_verify_detects_missing_directory(tmp_settings: Settings) -> None:
    root = tmp_settings.models_dir / "diarize"
    fake_weights(root)
    manage.register("diarize", root, tmp_settings)
    for p in root.rglob("*"):
        if p.is_file():
            p.unlink()
    (root / ".cache").rmdir()
    root.rmdir()
    with pytest.raises(ModelRegisterMismatch, match="directory missing"):
        manage.verify(tmp_settings)


def test_pull_never_persists_token(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    seen: dict[str, object] = {}

    def downloader(repo: str, local_dir: Path, token: str | None) -> tuple[Path, str]:
        seen.update(repo=repo, token=token)
        fake_weights(local_dir)
        return local_dir, "deadbeefcafe"

    rec = manage.pull("stt_en", tmp_settings, downloader=downloader)
    assert seen == {"repo": "mlx-community/whisper-large-v3-turbo", "token": TOKEN}
    assert rec.revision == "deadbeefcafe" and rec.conversion is None
    assert rec.local_path == tmp_settings.models_dir / "stt_en"
    register_text = manage.register_path(tmp_settings).read_text(encoding="utf-8")
    assert TOKEN not in register_text
    for p in list(tmp_settings.models_dir.rglob("*")) + list(tmp_settings.data_dir.rglob("*")):
        if p.is_file():
            assert TOKEN.encode() not in p.read_bytes(), p
    assert yaml.safe_load(register_text)["models"][0]["role"] == "stt_en"
    assert manage.verify(tmp_settings)[0].role == "stt_en"


def test_pull_without_token_and_conversion_recorded(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tokens: list[str | None] = []

    def downloader(repo: str, local_dir: Path, token: str | None) -> tuple[Path, str]:
        tokens.append(token)
        fake_weights(local_dir)
        return local_dir, "rev1"

    rec = manage.pull(
        "stt_ar",
        tmp_settings,
        downloader=downloader,
        converter=lambda p: "mlx 8-bit q_group_size=64",
    )
    assert tokens == [None] and rec.conversion == "mlx 8-bit q_group_size=64"
    assert rec.repo == "CohereLabs/cohere-transcribe-arabic-07-2026"
    with pytest.raises(ValueError, match="unknown model role"):
        manage.pull("llm", tmp_settings, downloader=downloader)


def test_register_rejects_empty_or_missing_directory(tmp_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        manage.register("stt_en", tmp_settings.models_dir / "nope", tmp_settings)
    empty = tmp_settings.models_dir / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        manage.register("stt_en", empty, tmp_settings)


def test_hashes_for_provenance(tmp_settings: Settings) -> None:
    assert manage.hashes_for_provenance(tmp_settings) == {}
    fake_weights(tmp_settings.models_dir / "stt_en")
    fake_weights(tmp_settings.models_dir / "stt_ar")
    manage.register("stt_en", tmp_settings.models_dir / "stt_en", tmp_settings)
    manage.register("stt_ar", tmp_settings.models_dir / "stt_ar", tmp_settings)
    first = manage.hashes_for_provenance(tmp_settings)
    assert set(first) == {"stt_ar", "stt_en"} and all(len(v) == 64 for v in first.values())
    assert manage.hashes_for_provenance(tmp_settings) == first  # deterministic
    (tmp_settings.models_dir / "stt_en" / "config.json").write_bytes(b"changed")
    manage.register("stt_en", tmp_settings.models_dir / "stt_en", tmp_settings)
    second = manage.hashes_for_provenance(tmp_settings)
    assert second["stt_en"] != first["stt_en"] and second["stt_ar"] == first["stt_ar"]
