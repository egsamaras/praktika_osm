.PHONY: dev test lint eval models doctor demo smoke word

# Development targets for a machine with uv. uv uses the project's .venv, which may be a symlink
# to a virtualenv kept outside the checkout (or set UV_PROJECT_ENVIRONMENT). A production host
# installs from uv.lock directly and needs none of these targets.

# Full development environment: diarisation, dev tools and the in-process MLX speech backends
# for Apple silicon (that extra carries platform markers and installs nothing on Linux).
dev:
	uv sync --frozen --extra mac --extra diarize --extra dev

# Offline test suite: no network, no model weights (tests needing weights are skipped).
test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

# Golden-set evaluation with the fake LLM; writes eval_report.md and fails on gate breach.
eval:
	uv run praktika eval --golden tests/fixtures/golden --llm fake

# One-time model mirror on a connected machine (HF_TOKEN from the environment only) and hash
# verification.
models:
	uv run praktika models pull stt_en
	uv run praktika models verify

doctor:
	uv run praktika doctor

# Ingest the English VTT fixture with the real Ollama and open the review page.
demo:
	uv run praktika ingest tests/fixtures/synthetic_en.vtt --type general --class internal \
		--lang en --roster tests/fixtures/roster_data_team.yaml
	uv run praktika serve

# Same path with the offline fake LLM into a throw-away data directory (see the README's "Two-minute smoke test").
# Like the README block it uses an empty env file inside SMOKE_DIR and a throw-away vault key, so
# it works the same on macOS and Linux, never reads your env file and never touches the login
# Keychain. The key lasts for this run only; the README block keeps one for a whole session.
SMOKE_DIR ?= $(CURDIR)/.praktika-smoke
smoke:
	mkdir -p "$(SMOKE_DIR)" && : > "$(SMOKE_DIR)/empty.env"
	PRAKTIKA_VAULT_KEY="$$(uv run --frozen python -c \
		'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
	PRAKTIKA_ENV_FILE="$(SMOKE_DIR)/empty.env" PRAKTIKA_DATA_DIR="$(SMOKE_DIR)" \
	PRAKTIKA_LLM_PROVIDER=fake PRAKTIKA_PILOT_SMOKE=true \
		uv run --frozen praktika ingest tests/fixtures/synthetic_en.vtt --type general \
		--class internal --lang en --roster tests/fixtures/roster_data_team.yaml \
		--notified --no-objections --method chat --teams-transcription-started \
		--purpose "Smoke test of the notetaker pipeline with synthetic data" --ack-all-scope

word:  ## Word copies of every docs/*.md, black and white, next to each source file
	@set -e; for f in docs/*.md; do uv run python scripts/md_to_docx.py "$$f"; done
