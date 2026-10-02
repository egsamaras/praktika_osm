# Developing Praktika

Praktika is developed on Apple silicon Macs and tested on Linux. The same code runs on both; the
differences are which speech backend runs, where the vault key lives and which scheduler runs the
retention timers.

## Prerequisites

* [uv](https://docs.astral.sh/uv/). It installs Python 3.12 if you do not have it; the project
  requires `>=3.12,<3.13`.
* `ffmpeg`, for audio ingest and for the tests that convert audio.
* `git`.
* For real drafting (not needed for the tests): [Ollama](https://ollama.com) with
  `ollama pull qwen2.5:14b`.

On macOS: `brew install uv ffmpeg`. On Ubuntu: `apt-get install ffmpeg git` and uv from its
installer or release archive.

## Set up

Praktika is developed and run from a git checkout installed in editable mode; it is not published
on PyPI. On an Apple silicon Mac, which also gets the in-process MLX speech-to-text:

```bash
git clone https://github.com/egsamaras/praktika_osm.git
cd praktika_osm
uv sync --frozen --extra dev --extra mac
```

On Linux, or on a Mac without in-process speech, use `uv sync --frozen --extra dev` for the last
line instead. `uv sync` makes the environment match exactly the extras you name and removes the
others, so a later plain `uv sync --frozen --extra dev` on a Mac removes the MLX packages again.
On x86_64 Linux the environment takes about 6.6 GB, because PyPI's `torch` wheel there brings
NVIDIA's CUDA libraries; on macOS it is much smaller (about 0.5 GB without the `mac` extra).

`make dev` runs `uv sync --frozen --extra mac --extra diarize --extra dev`. The extras are:

| Extra | What it adds | Where |
|---|---|---|
| `dev` | pytest, pytest-cov, ruff, respx, hypothesis, freezegun, jiwer | everywhere |
| `mac` | MLX, mlx-whisper and mlx-audio for in-process speech-to-text | macOS only (platform markers: installs nothing elsewhere) |
| `diarize` | pyannote.audio and torchaudio | optional, both platforms |
| `cuda` | faster-whisper and CTranslate2, an in-process speech backend for x86 CUDA hosts | development only; on aarch64 it runs on the CPU |
| `graph` | `msal`, for a future Microsoft Graph transcript poller | not used yet |

`uv.lock` pins every version. Use `--frozen` so that uv installs exactly what is locked; if you
change a dependency, run `uv lock` and commit the lock file with the change.

### Keep the virtual environment out of cloud-synced folders

If your checkout lives in a folder synced by a cloud service (iCloud Drive's Desktop and Documents,
OneDrive, Dropbox and the like), keep the virtual environment outside it. Some sync clients set
the macOS "hidden" flag on files they manage, and Python 3.12 skips hidden `.pth` files, which
silently breaks the editable install: the `praktika` command then fails to import its own package.
Syncing a virtual environment also uploads gigabytes for nothing.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/praktika-osm"
uv sync --frozen --extra dev --extra mac
```

Either keep `UV_PROJECT_ENVIRONMENT` exported, or create the environment once with it and then
link it into the checkout (`ln -s "$HOME/.venvs/praktika-osm" .venv`); uv follows the link and
`.gitignore` ignores `.venv`. The same applies to data: Praktika refuses to write exports into
iCloud-synced folders unless you pass `--force`, and its default data directory
(`~/Library/Application Support/Praktika`) is not synced.

## Everyday commands

| Command | What it does |
|---|---|
| `make test` | `uv run pytest`: offline; tests that need model weights are skipped |
| `make lint` | `ruff check` and `ruff format --check` on `src` and `tests` |
| `make eval` | The golden set with recorded model outputs; must print `Gate: PASS` |
| `make smoke` | Ingests the synthetic VTT with the fake LLM into `.praktika-smoke/`, with an empty env file and a throwaway vault key there |
| `make doctor` | Readiness checks for this machine |
| `make demo` | Ingests the synthetic VTT with the real Ollama model, then starts the review page |
| `make models` | Mirrors the English speech model (MLX) and verifies its hashes |
| `make word` | Word copies of every `docs/*.md`, next to each source file |

On Linux, `make demo` needs `PRAKTIKA_VAULT_KEY` exported (see the README's smoke test for how to
generate one); on a Mac the key is created in the login Keychain on first use.

## Tests

The test suite is offline: no network, no model weights, no GPU. Every fixture is synthetic.

* Tests that need real weights are marked `models` and skip unless `PRAKTIKA_MODELS_DIR` holds them.
  A `slow` marker is declared for long-running tests; no test uses it yet. Markers are strict.
* HTTP clients are tested with `respx` and `httpx.MockTransport`; the LLM pipeline with the
  deterministic fake model and with recorded golden playback.
* A few tests skip when `node` is not installed (they check the review page's JavaScript), and one
  needs `/proc` (Linux).
* `tests/test_scope_consent.py::test_no_skip_flag_exists` fails if any option that could skip the
  consent gate appears; `tests/test_policy.py::test_no_analytics_fields` fails if per-person
  analytics fields appear in `src/`.

Run one file or one test with `uv run pytest tests/test_verify.py -k quote`.

### The Linux test image

`docker/linux-test.Dockerfile` builds an Ubuntu 24.04 aarch64 image with the distribution's Python
3.12, uv from its pinned release archive, and the project installed from `uv.lock` with the `dev`
extra and without the `mac` extra, running as a non-root user. It checks packaging and code on
Linux, not speed, and has no GPU.

```bash
docker build --platform linux/arm64 -f docker/linux-test.Dockerfile -t praktika-linux-test .
```

The image leaves out the synthetic speech recording, `tests/fixtures/synthetic_meeting.wav`, to
stay small; the `docker run` command in the Dockerfile's header mounts it read-only so that the
voice-activity and routing tests can read it.
On an x86_64 machine the build needs arm64 emulation and is slow. CI runs the suite natively on
`ubuntu-latest` (x86_64) instead; see `.github/workflows/ci.yml`.

## What differs on macOS

* **Speech-to-text.** The default `PRAKTIKA_STT_EN=mlx_whisper` runs Whisper large-v3-turbo in
  process on MLX (`stt/mlx_whisper_backend.py`) and needs the `mac` extra and the weights:
  `uv run praktika models pull stt_en` mirrors `mlx-community/whisper-large-v3-turbo` into
  `~/praktika-models/stt_en` and records every file's hash in `~/praktika-models/models.yaml`.
* **Arabic, for development.** The Arabic path is off (`PRAKTIKA_STT_AR=none`).
  `PRAKTIKA_STT_AR=mlx_whisper_full` (weights: `models pull stt_ar_full`) or
  `PRAKTIKA_STT_AR=mlx_cohere` (weights: `models pull stt_ar`, a gated repository that needs
  `HF_TOKEN` from an account that has accepted its terms) switch it on.
* **Vault key.** With `PRAKTIKA_VAULT_KEY` unset in local mode, the key is created and kept in the
  login Keychain (generic password `praktika-vault`). Linux has no such fallback.
* **Retention.** `praktika retention install` writes a launchd agent to `~/Library/LaunchAgents`
  instead of the systemd timer.
* **Data directory.** `~/Library/Application Support/Praktika`, with the default env file `.env`
  inside it.
* **Disk encryption.** `doctor` checks FileVault (`fdesetup status`) on macOS and a dm-crypt
  mapping on Linux.
* **Live capture.** `praktika start --source mic` records the default microphone (your terminal
  needs microphone permission). `--source sck` expects a signed system-audio capture helper that
  is not part of this repository.

## Layout

```
src/praktika/          the package (see docs/ARCHITECTURE.md for a module map)
prompts/v1/            the prompt set; its hash is pinned in src/praktika/llm/prompts.py
templates/render/      Jinja2 templates for Markdown export, one per minutes template
glossary.yaml          example glossary (terms, variants, speech-to-text misrenderings)
tests/                 the test suite; tests/fixtures/ holds synthetic data and the golden set
scripts/md_to_docx.py  converts a docs/*.md file to a plain Word document
docker/                the Linux test image
```

## Conventions

* Every module starts with a docstring that states its contract.
* No `print()` in `src/` (ruff rule `T20`): use the CLI console or the structured logger.
* Any outbound HTTP client comes from `Settings.http_client()`, so that the egress allow-list
  applies per request. A new URL setting goes into `config._URL_FIELDS` so that it is validated
  at start-up.
* Subprocess calls use fixed executables with explicit argument lists and carry a
  `# noqa: S603` with a one-line justification.
* Files holding meeting content are created with mode 0600 inside 0700 directories.
* A state change, a refusal or anything a reviewer or operator does on a meeting is audited through
  `AuditLog.append`; model calls are audited without content.
* A change to `prompts/` changes the prompt-set hash. Update `PINNED_SHA256` in
  `src/praktika/llm/prompts.py` with the value `praktika.llm.prompts.version_sha256` computes, run
  `make eval`, and say why in `CHANGELOG.md`.
* Synthetic data only, everywhere: tests, fixtures, the golden set, issues and pull requests. See
  [CONTRIBUTING.md](../CONTRIBUTING.md).

## Regenerating fixtures

* `uv run python tests/fixtures/make_vtt_fixtures.py` regenerates the synthetic Teams transcripts
  (`synthetic_en.vtt`, `synthetic_ar_mixed.vtt`) and the recap document
  (`synthetic_recap.docx`). The output is byte-for-byte stable: the `.docx` zip entries carry a
  fixed timestamp.
* `uv run python tests/fixtures/make_fixture.py tests/fixtures` regenerates the synthetic speech
  recording `synthetic_meeting.wav` (about two minutes of a fictional "Data team weekly" meeting
  at Acme Bank, with three invented speakers, one of them speaking Arabic) and its ground truth
  `synthetic_meeting.truth.json`. It needs macOS, for the `say` voices it names (they must be
  installed), and ffmpeg. The recording depends on the voices' versions, so it is not
  byte-for-byte stable: after regenerating, run the whole suite, because the voice-activity,
  routing and speech-backend tests use time slices of it. `.gitignore` excludes recordings; this
  file and the short tone `tone_16k.wav` are the only exceptions.
