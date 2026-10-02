# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-02

First public release. Pilot-grade: the controls are in code and tested; the tool has not been
validated on real meetings at scale. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

### Added

* `praktika` command-line interface and FastAPI review page.
* Consent and scope gate with no skip flag; spoken script and chat notice in English and Arabic.
  Pilot mode (on by default) accepts Internal meetings only and refuses board, HR, customer,
  regulator, legally privileged and foreign-hosted meetings.
* Ingest of Microsoft Teams transcripts (`.vtt`, recap `.docx`) and recordings (`.wav`, `.m4a`,
  `.mp3`, `.mp4`), and live microphone capture.
* English speech-to-text with Whisper large-v3-turbo, in process on Apple silicon (MLX) or over
  HTTP from a vLLM server; voice activity detection with silero; an optional Arabic path, off by
  default; optional diarisation, off by default.
* Identifier tokenisation (e-mail, IBAN, account, card, Bahraini CPR, Saudi iqama, phone, amounts
  near a name) with a Fernet-encrypted vault.
* Map-reduce drafting through Ollama or an OpenAI-compatible server, JSON-schema constrained, with
  `general`, `mancom` and `one_to_one` templates.
* Deterministic verifier for citations, quotes, numbers, names, speakers and instruction-like
  segments.
* Review page with flags, per-item verdicts and reason codes, speaker mapping, section
  regeneration, approval and discard.
* Markdown and DOCX export of approved minutes, with classification and provenance.
* Hash-chained audit log; retention timers with legal hold and an hourly scheduler (systemd or
  launchd); data-subject request commands; full-text search over approved minutes.
* Egress allow-list for every outbound request; per-meeting run lock; model register with
  per-file hashes; `praktika doctor`.
* Service mode with OIDC token validation and four roles (tested with synthetic tokens only).
* Golden-set evaluation (`praktika eval`) with six synthetic meetings and rollout gates.
* Documentation: architecture, Linux GPU server deployment, control catalogue, evaluation,
  development and limitations; contribution guidelines, a code of conduct, a security policy, and
  issue and pull request templates.

<!-- These two links work only once the v0.1.0 tag has been pushed and a GitHub release created
for it; until then they return 404. Replace egsamaras at the same time. -->
[Unreleased]: https://github.com/egsamaras/praktika_osm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/egsamaras/praktika_osm/releases/tag/v0.1.0
