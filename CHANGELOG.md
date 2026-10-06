# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

* A single-word glossary misrendering now matches only exactly, never fuzzily: one letter away is
  where names and ordinary words are ("Deepia" turned "Deepika" into "DPIA"). List every spelling
  you see. Multi-word misrenderings are still matched fuzzily.
* `praktika dsar find|export|delete` also match a meeting's organiser (UPN) and a full name of two
  words or more in its title (as whole words; a single word such as "May" never matches a title).
  `dsar find`, and the confirmation of `dsar delete`, show why each meeting matched.
* An unknown key id in a token triggers at most one JWKS fetch a minute (timed with a clock that
  never steps back), and a failed fetch is remembered for that minute, so a stream of made-up key
  ids can no longer make the server hammer the identity provider. A rotated key is picked up
  within the minute.
* An RSA key the identity provider publishes for encryption (`"use": "enc"`) no longer verifies
  tokens; only signing keys do.
* `glossary.load` refuses more single-word misrenderings: Q1 to Q4, Dale, Del, Jenny, Pratik,
  Pratika, Practice, Practices, Practical, Team's and Whisperer. A glossary of your own that lists
  any of them stops ingest with an error until you replace it with a multi-word pattern; the
  example glossary of 0.1.0 listed "Pratika".

### Fixed

* Glossary normalisation changed everyday speech in the transcript, which is the evidence behind
  every citation: "Practice makes perfect" became "Praktika makes perfect", an opening quote next
  to a corrected word was dropped, "Bahrain, dinner" was read as one misrendering and "S A R A H"
  became "SAR A H". The example glossary drops "Pratika", "Riyadh al", "Bahrain dinner" and
  "Bahraini dinner", which rewrote a name, a district and a real dinner.
* Markdown export puts each citation on its own line; several citations of one topic, question
  or risk ran together. An item without citations no longer adds a blank line (which made
  Markdown render the whole list loose), and a topic's citations follow an "Evidence:" line
  instead of nesting under the topic's last key point.
* Title warnings for pilot exclusions match a keyword only at the start of a word ("auditor" and
  "hr" must also end at one): "Dashboard review", "Onboarding plan", "Town hall in the
  auditorium" and "3hr workshop" no longer warn, while file-name titles ("Board_Meeting",
  "boardmeeting", "Q4Board", "BoardMeeting") and Arabic prefixes attached to a word still do.
  "HR" is now recognised next to punctuation and in camelCase ("HR: policy", "HR-Finance",
  "HRMeeting"), and common Arabic spelling variants now warn as well.
* `praktika doctor` (plain and `--json`) and `praktika config show` printed a URL setting with any
  user name and password in it, and at `--log-level INFO` every command logged each HTTP
  request's full URL; the user name and password of any URL are now replaced by `***` in their
  output and in every log line.
* When the review server could not fetch the identity provider's signing keys, the 401 sent to the
  caller, and the access-denied log line, named the JWKS URL with any user name and password in
  it; both now give only a short reason (the HTTP status, the error type, or a document with no
  key list), and the full cause
  goes to the operator log with any password masked. A JWKS fetch the allow-list refused gave a
  409 naming the host and the whole allow-list; it is now a 401 like any other failed fetch.
* Sign-in failed for every token when the identity provider also published a key that is not an
  RSA key (an EC key, for example); such keys are now skipped. A malformed JWKS document, or a
  token with a non-ASCII byte or deeply nested JSON, gave an unaudited 500; all are now refused
  with a 401 and audited.
* An egress error printed the whole URL of the setting, including any password in it; it now
  names the setting and the host. A setting that fails validation is reported by name and reason,
  without the value given,
  and a malformed `PRAKTIKA_ALLOWED_HOSTS` is reported as itself rather than as an egress error of
  another setting.
* An allow-list entry naming an IPv6 address (`fd00::10` or `[fd00::10]`, short or long form) now
  matches a URL to that address; before, only a glob such as `*fd00::*` could, and globs still do.
* The Graph error hint for `DeltaFilterNotAllowed` told administrators to change the application
  access policy; a filter on a delta link is a defect in the poller. A missing access policy has
  its own code, `ApplicationAccessPolicyMissing`, and the tenant hints name `-Identity Global`.

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

[Unreleased]: https://github.com/egsamaras/praktika_osm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/egsamaras/praktika_osm/releases/tag/v0.1.0
