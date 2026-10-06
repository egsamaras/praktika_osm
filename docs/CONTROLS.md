# Privacy and security controls

This is Praktika's control catalogue, C-01 to C-39. The identifiers of most core controls appear in
code comments and docstrings, so `grep -rn "C-06" src/` finds the code behind them; for the other
controls, the table names the modules. For each control the table says what it requires, what the
code does and where, and what remains the responsibility of the organisation that deploys
Praktika (the **deployer**).

The catalogue is a technical starting point, not a compliance framework. It does not tell you
which laws apply to you or whether your deployment meets them; see the disclaimer in the
[README](../README.md#disclaimer).

Status values:

* **Built**: implemented in code and covered by tests.
* **Partly built**: the code covers part of the control; the rest is listed.
* **Not built**: no code yet; listed in [LIMITATIONS.md](LIMITATIONS.md).
* **Deployer**: an organisational measure that code cannot provide.

Paths are relative to `src/praktika/` unless they start with `tests/`.

## Core controls

These are the controls the code is built around. Most are enforced in code; several also need the
deployer to do something.

| ID | Control | What the code does, and where | Status | Deployer's part |
|---|---|---|---|---|
| C-01 | On-premises inference only: no external speech, LLM or embedding endpoint can be reached | Every configured URL (`llm_base_url`, `stt_http_url`, OIDC issuer and JWKS) must match `PRAKTIKA_ALLOWED_HOSTS` or start-up fails with `EgressError`, whose message names the setting and host, never the URL (which may carry a password); patterns without a literal host label (`*`) are refused; every outbound HTTP client is built by `Settings.http_client()`, whose `AllowListTransport` checks the host on each request; `HF_HUB_OFFLINE=1` is forced at import (only `praktika models pull` lifts it, for its own download). On macOS, exports into iCloud-synced folders are refused unless `--force`. `config.py`, `llm/ollama_client.py`, `llm/openai_compat_client.py`, `stt/http_backend.py`, `identity.py`, `cli/export_paths.py`; `tests/test_config_no_egress.py` | Built | Block outbound traffic at the network for the host, Ollama and the speech container: Praktika's allow-list governs Praktika's own process only. Keep the allow-list to fully qualified names |
| C-02 | No voiceprints: no speaker enrolment, no cross-meeting matching; diarisation off by default | `PRAKTIKA_DIARIZE` defaults to false. A diarisation `Turn` holds start, end and an anonymous label only; the pyannote backend reads labels and times and never stores the embeddings it computes. `diarize/base.py`, `diarize/pyannote_backend.py`; `tests/test_assign.py` | Built | Decide whether diarisation may be used, after a data-protection assessment (C-22) |
| C-03 | Consent and notice gate: nothing is stored until the organiser confirms notice given, no objections, method, purpose and every scope item; no skip flag | `consent.gate` is the only way to obtain a `ConsentRecord`. It refuses unless attendees were notified and nobody objected, the scope checklist is fully confirmed and the classification is allowed. The purpose (10 to 500 characters), the notice method and the script version are stored. A refusal is audited as `scope.refused` with the neutral reason "recording not used", never naming an objector. Missing answers are asked for in a terminal or refused (exit 2) without one. The classification is not asked for: it comes from `--class` and defaults to `internal`. The spoken script and the chat notice exist in English and Arabic (`praktika consent-script`). `consent.py`, `scope.py`, `cli/gate_prompts.py`; `tests/test_scope_consent.py::test_no_skip_flag_exists` | Built | The attestation is procedural: nothing verifies that notice was really given. Approve the wording, the privacy notice and the objection procedure, and make the wording describe your retention settings and who can see drafts (see [the consent script](#consent-script-and-chat-notice) and [LIMITATIONS.md](LIMITATIONS.md#scope-and-consent)) |
| C-04 | Visible recording state and a kill switch that purges in-flight audio | Live capture (`praktika start`) shows a status line with elapsed time and input levels and warns when the input is silent; there is no silent mode. `praktika abort <id>` from any shell stops the capture and overwrites and unlinks the meeting's audio. `audio/capture.py`, `cli/meetings.py`, `cli/meeting_ops.py`; `tests/test_capture.py` | Built | — |
| C-05 | Raw audio deleted automatically; no audio export | Converted audio is deleted by the first retention run after the minutes are approved or discarded, and in any case by the first run after the classification's maximum has passed since conversion (24 h Internal, 72 h Confidential by default); with a maximum of 0 (Restricted by default) it is deleted by the first run after transcription. Retention runs happen at the start of every command that opens the meeting store (C-14 lists them), once when `serve` starts, with `praktika retention run`, and hourly once `praktika retention install` has set up the scheduler; without the scheduler the deadlines slip until a command runs. Deletion overwrites the file with zeros, fsyncs and unlinks it before the database row is updated, and orphaned WAV files are swept. There is no audio export command or route; the review page plays slices of retained audio only. `retention.py`, `retention_files.py`, `store/artefacts.py`, `ingest/audio_file.py`; `tests/test_retention.py`, `tests/test_store_retention.py` | Built | Install the retention scheduler on every host. Make the consent wording and the retention settings agree (see [the consent script](#consent-script-and-chat-notice) and [LIMITATIONS.md](LIMITATIONS.md#scope-and-consent)). Delete the source file you ingested; Praktika does not |
| C-06 | Identifiers tokenised before any model sees the text; reversible only in approved, non-restricted minutes | E-mail addresses, IBANs (mod-97 checked), account numbers next to a context word, card numbers (Luhn checked), Bahraini CPR and Saudi iqama or national ID numbers next to a context word, Bahraini, Saudi and UK phone numbers, and amounts spoken near a person's name are replaced by stable tokens such as `«IBAN_1»`. Arabic-Indic digits are normalised first. The token vault is Fernet-encrypted. The pipeline refuses an unredacted transcript. De-tokenisation (`export --detokenise`) is allowed only for approved, non-restricted minutes. `redact/`, `llm/pipeline.py`, `policy.py`; `tests/test_tokenise.py` | Built | Hold the vault key (`PRAKTIKA_VAULT_KEY`) in a secrets manager; there is no escrow or rotation (see [LIMITATIONS.md](LIMITATIONS.md)) |
| C-07 | Human review before any distribution; every item cites the transcript; no autonomous actions | Every decision, action, open question, risk and figure must cite segment ids. The deterministic verifier checks the citations and quotes of decisions, actions, open questions and risks, the numbers and names in the minutes, speakers, audio confidence, MNPI keywords and raw identifiers, and moves uncited decisions and actions into priority-1 flags with their full text so that a reviewer can restore them. The exception is the management-committee "Figures mentioned" list: it is copied from the model's findings, its segment ids are not resolved, its numbers are not checked, and the review page does not show it (it appears in the exports). Items are accepted, modified or rejected with a reason code. Approval is refused while a priority-1 flag is open. Export is refused before approval; a draft export needs both the caller (`--allow-draft`) and the deployment (`PRAKTIKA_ALLOW_DRAFT_EXPORT=true`) and is always refused in pilot mode. Praktika sends nothing to anyone. `llm/verify.py`, `llm/verify_flags.py`, `llm/quotes.py`, `server_review.py`, `cli/review_cmd.py`, `policy.py`; `tests/test_verify.py`, `tests/test_server.py` | Partly built | Train reviewers; the review can be done badly (see [LIMITATIONS.md](LIMITATIONS.md)). Check management-committee figures against the transcript by hand |
| C-08 | Named identity for every action, encryption at rest, tamper-evident audit | Every action is attributed to an `Identity` with its source (`session`, `local`, `oidc`). Every command that opens the meeting store forces the data directory to 0700, and the database, audit log and exports are 0600. The vault key is held outside the database. The audit log is hash-chained, written with `fsync`, and cross-checked against its head file and the database's audit table by `praktika audit verify`. `doctor` checks disk encryption (dm-crypt on Linux, FileVault on macOS). `identity.py`, `server_auth.py`, `audit.py`, `store/db.py`, `cli/doctor.py`; `tests/test_identity.py`, `tests/test_audit.py`, `tests/test_audit_concurrency.py`, `tests/test_server_security.py` | Built | Encrypt the volume that holds the data directory. Provide single sign-on for a shared deployment (C-20). Ship the audit stream off the host |
| C-09 | Classification on every meeting | Each meeting is `internal`, `confidential` or `restricted`. Pilot mode accepts `internal` only. Outside pilot mode, local mode still accepts `internal` only, and `confidential` and `restricted` need service mode; `restricted` minutes are never indexed for search. Exports carry the classification in header and footer. `policy.py`, `store/search.py`, `render/`; `tests/test_policy.py` | Built | Set the classification rules for your organisation |
| C-10 | Scope exclusions enforced in code | With `PRAKTIKA_PILOT=true` (the default), board and board sub-committee meetings, HR, customer, regulator or auditor, legally privileged and foreign-hosted meetings are refused when the organiser tags them so (`--tag board`, `--tag hr`, ... or `--foreign-hosted`). In every mode, pilot or not, every scope checklist item (not a board meeting, not HR, and so on) must be confirmed at the gate (C-03); only the tag-based refusals depend on pilot mode. A title that suggests an exclusion produces a warning. `--external-participants` records the tag `external` without refusing. `scope.py`, `cli/gate_prompts.py`; `tests/test_scope_consent.py` | Built | Decide your own exclusions; the list is fixed in `scope.py` |
| C-11 | Governance before use: a documented use case, risk assessment, data-protection impact assessment, security assessment, an entry in the organisation's AI inventory, and approval with exit criteria | — | Deployer | All of it |
| C-12 | Privacy notice, participant rights and data-subject requests | `praktika dsar find`, `dsar export` and `dsar delete` locate, export and delete a participant's meetings: by roster name, alias or UPN, transcript speaker, the meeting's organiser (UPN), or a full name of two words or more in the title (as whole words; a single word never matches a title). `dsar find` shows why each meeting matched. A deletion is refused while any of those meetings is on legal hold. `cli/ops.py`, `store/` | Partly built | Publish the privacy notice and run the rights process |
| C-13 | Reproducibility: prompts, glossary and models versioned and hashed per run | Prompts are versioned under `prompts/<version>/` and their SHA-256 is pinned in `llm/prompts.py` (`doctor` warns when they differ). The glossary's hash is recorded. `models.yaml` records each model's repository, revision, licence and the SHA-256 of every file, and `praktika models verify` re-hashes them. Every minutes version carries a `Provenance` record: model name and digest, prompt version and hash, glossary hash, template, git commit, transcript hash, speech engines and model hashes. Every model call is audited as `llm.call` with the prompt hash and timings, never the content. `llm/prompts.py`, `models_registry/`, `models/minutes.py`, `llm/base.py`; the golden set in `tests/fixtures/golden/` | Built | Keep the register and the prompt pin under change control |
| C-14 | Retention timers with legal hold and deletion receipts | Timers per classification for audio (C-05); for the transcript and its vault, `PRAKTIKA_RETENTION_TRANSCRIPT_DAYS` after approval (14 days by default, 7 for Restricted) and in any case 60 days after the transcript was first stored (30 for Restricted), a maximum fixed in `retention.TRANSCRIPT_MAX_DAYS`; for unapproved drafts, `PRAKTIKA_RETENTION_DRAFT_DAYS` after approval or after the latest draft (30 days by default, 14 for Restricted). Approved minutes and exports have no timer. Each deletion is audited as `retention.deleted`, and a failure as `retention.failed`. `praktika hold set` blocks every timer for a meeting. Timers run at the start of every command that opens the meeting store (`ingest`, `start`, `transcribe`, `generate`, `abort`, `approve`, `export`, `search`, `actions`, `hold`, `dsar`), once when `serve` starts, with `praktika retention run`, and hourly from the scheduler `praktika retention install` writes (systemd on Linux, launchd on macOS). `retention.py`, `store/artefacts.py`, `cli/ops.py`; `tests/test_retention.py::test_legal_hold_blocks_all` | Built | Approve the retention schedule; install the scheduler; keep backups from outliving it; put exports under your own records retention |
| C-15 | No analytics on individuals (talk time, sentiment, attendance scoring) | No such field exists; a test scans `src/` for them. `tests/test_policy.py::test_no_analytics_fields` | Built | Do not add them |
| C-16 | Model licence due diligence recorded | The register records a licence for every model; `praktika models register --licence` sets it for weights brought in by other means. `models_registry/manage.py` | Partly built | Check each model's licence and terms before use (see the README's model table) |
| C-17 | Staff transparency: notice to staff, voluntary participation, opt-out without detriment | — | Deployer | All of it |

## Before wider use

These controls matter once Praktika moves beyond a small trial group, takes confidential
meetings, records guests, captures live audio or crosses a border.

| ID | Control | What the code does, and where | Status | Deployer's part |
|---|---|---|---|---|
| C-18 | Full data-protection impact assessment, with a legitimate-interests assessment where that is the lawful basis; any registration or notification your data-protection law requires | — | Deployer | All of it |
| C-19 | Conferencing-platform consent: the platform's own recording-consent policy for organisers, a custom banner and a privacy-notice link; consent evidence retained | The gate records whether the platform's own transcription was started (`--teams-transcription-started`) | Deployer | Configure the platform |
| C-20 | Server deployment behind an authenticating gateway: OIDC token validation (never trusted headers), roles from directory groups, append-only audit, SIEM shipping, secrets from a secrets manager | Service mode refuses to start without `PRAKTIKA_IDENTITY_PROVIDER=oidc`. `OidcIdentity` validates RS256 tokens against the issuer's JWKS (issuer, audience, expiry) and maps four groups or app roles to roles; `X-Auth-*` style headers are never read. With `PRAKTIKA_AUDIT_SINK=stdout` a copy of every audit line goes to `audit-forward.jsonl` for a log shipper. `identity.py`, `server_auth.py`, `server.py`, `audit.py`; `tests/test_identity.py::test_headers_are_never_trusted`, `tests/test_server.py::test_service_mode_network_bind_needs_oidc` | Partly built | Run the gateway, the identity provider and the log shipping. OIDC has been tested with synthetic tokens only |
| C-21 | Any capture client on a staff laptop is code-signed, reviewed and allow-listed, shows a recording indicator, never starts on its own and captures only the call | `praktika start --source sck` refuses a helper that does not carry a Developer ID signature from the pinned Team ID (`PRAKTIKA_CAPTURE_TEAM_ID`). The helper itself is not in this repository. `audio/signing.py`, `cli/meetings.py` | Partly built | Build, sign and review any capture helper |
| C-22 | A written data-protection position on transient diarisation embeddings before diarisation is used on real meetings | Diarisation is off by default (C-02) | Deployer | All of it |
| C-23 | Cross-border assessment where participants or the hosting entity are in another jurisdiction: lawful basis, transfer rules, local hosting if required | `--foreign-hosted` (tag `foreign_hosted`) is refused in pilot mode | Deployer | All of it |
| C-24 | A committee-minutes mode that meets corporate-governance requirements (attendance, absences, discussion, recommendations, decisions, dissent, implementation status), with the company secretary as author of record | Templates `general`, `mancom` and `one_to_one` only | Not built | Decide whether committee minutes may be drafted this way at all |
| C-25 | Records integration: approved minutes filed in the records system with its retention, classification metadata and data-loss prevention on export locations | Exports are written 0600 to `<data_dir>/exports/` and carry the classification | Not built | Filing and records retention |
| C-26 | Independent validation (word error rate by accent and gender, attribution error, omission rate, fabrication rate, injection resilience), monitoring tolerances and a human-in-the-loop threshold register | The synthetic golden set and its gates ([EVALUATION.md](EVALUATION.md)) | Not built | Validation on consented real meetings |
| C-27 | HR policy, staff communication (transparency, opt-out without detriment, no performance use) and training for organisers and reviewers | — | Deployer | All of it |
| C-28 | Guests and vendors: invitation text with a notice link, explicit agreement captured, refusal path tested | `--external-participants` records the tag `external` | Deployer | All of it |
| C-29 | Incident runbook (unconsented capture, a transcript sent to the wrong person, device loss), with the notification deadlines that apply to you | — | Deployer | All of it |
| C-30 | Material-change control: hosted models, recording download, a new jurisdiction, new meeting types, sending features or per-person analytics need fresh approval | — | Deployer | All of it |
| C-31 | Microsoft Graph access, when built: application permissions with a certificate credential, an access policy scoped to named organisers, egress only to the Microsoft sign-in and Graph hosts, every read audited, a separate process with its own allow-list | The protocol and the error vocabulary only, with the tenant change each error calls for. `ingest/graph_stub.py`; `tests/test_graph_stub.py` | Not built | Tenant configuration |

## Recommended

| ID | Control | What the code does, and where | Status | Deployer's part |
|---|---|---|---|---|
| C-32 | Arabic-first consent flow, an Arabic minutes template and a Gulf-dialect evaluation set | The consent script, chat notice and scope prompts exist in Arabic; two golden meetings are mixed Arabic and English. No Arabic minutes template | Partly built | — |
| C-33 | Purpose statement stored per meeting | The organiser's purpose is part of the consent record and audited with `consent.recorded`. `consent.py` | Built | — |
| C-34 | Retention differing by classification; deletion receipts in the audit log | See C-05 and C-14 | Built | — |
| C-35 | Periodic data-protection and compliance sample review of approved minutes for sensitive data and inside information | The verifier raises `possible_mnpi` and `identifier_detected` flags. `llm/verify_flags.py` | Deployer | Run the review |
| C-36 | Watermarked exports and reader-mode viewing for Restricted minutes; a searchable archive only for non-restricted minutes, scoped per owner | Restricted minutes are never indexed; search covers approved minutes only; drafts carry a "DRAFT — NOT APPROVED" banner and every export carries its classification. A Restricted watermark and a reader mode are not built. `store/search.py`, `render/` | Partly built | — |
| C-37 | Participant self-service: see the meetings you took part in, ask for an attribution correction | — | Not built | — |
| C-38 | Legal review of the translated consent script and notices in the local language | — | Deployer | All of it |
| C-39 | Reviewer concordance metrics (edit rate per section, restore rate of removed items) reported regularly | Every verdict, edit and restore is stored and audited (`review.item`); no report is built | Not built | — |

## Consent script and chat notice

`praktika consent-script` prints these texts, and `praktika ingest` and `praktika start` record
their version (`2026-10-02`) in every consent record. They are written to be organisation-neutral;
adapt them to your organisation and have them approved before use. Every statement in them holds
with the default settings (pilot mode, so Internal meetings only, in local mode) once the hourly
retention scheduler is installed (`praktika retention install`): the first retention run after the
minutes are approved or discarded deletes the audio, and the first run 24 hours after Praktika
converted the recording deletes it otherwise; the transcript and its vault are deleted
`PRAKTIKA_RETENTION_TRANSCRIPT_DAYS` after approval (default 14 days; 7 for Restricted) and at the
latest 60 days after the transcript was first saved (30 for Restricted; fixed in
`retention.TRANSCRIPT_MAX_DAYS`); a legal hold (`praktika hold`) stops every timer; and nothing is
exported before approval. Without the scheduler the timers run only when a Praktika command opens
the meeting store, so the deadlines can slip. Change the wording, and `SCRIPT_VERSION` in
`src/praktika/consent.py`, if you change the retention settings, run without the scheduler, or run
in service mode, where a secretary can read, edit and approve the minutes of meetings that are not
private and the DPO and admin roles can read every draft. See also
[LIMITATIONS.md](LIMITATIONS.md#scope-and-consent).

**Spoken by the organiser at the start of the meeting (English):**

> Before we start: this meeting is being recorded and transcribed by our internal AI notetaker, which runs only on our own systems, to produce the minutes. The recording is deleted after the minutes are approved or discarded, and within about a day in any case. The transcript is deleted within two weeks of the minutes being approved, and within 60 days in any case. A legal hold can require us to keep them longer. I check and approve the minutes before they are shared. Nothing is used to evaluate individuals. If anyone prefers not to be recorded, say so now or message me privately and we will take notes manually instead. Does everyone agree to proceed?

**Chat notice (English, at most 200 characters):**

> Our internal AI notetaker is recording and transcribing for minutes, on our systems only. Audio deleted within about a day, transcript within 60 days, unless on legal hold. Object via the organiser.

**Spoken script (Arabic):**

> قبل أن نبدأ: يتم تسجيل هذا الاجتماع وتفريغه نصياً بواسطة أداة تدوين المحاضر الداخلية لدينا بالذكاء الاصطناعي، والتي تعمل فقط على أنظمتنا الخاصة، وذلك لإعداد محضر الاجتماع. يُحذف التسجيل الصوتي بعد اعتماد المحضر أو إلغائه، وفي جميع الأحوال خلال يوم تقريباً. ويُحذف النص المفرَّغ خلال أسبوعين من اعتماد المحضر، وفي جميع الأحوال خلال ستين يوماً. وقد يقتضي حفظ قانوني الاحتفاظ بهما مدة أطول. وأقوم بمراجعة المحضر واعتماده قبل مشاركته. لا تُستخدم هذه المعلومات لتقييم أداء الأفراد. إذا كان أي منكم يفضّل عدم تسجيله، فليُبلغني الآن أو عبر رسالة خاصة وسنقوم بتدوين الملاحظات يدوياً. هل يوافق الجميع على المتابعة؟

**Chat notice (Arabic, at most 200 characters):**

> أداة تدوين المحاضر الداخلية لدينا تسجّل الاجتماع وتفرّغه لإعداد المحضر، على أنظمتنا فقط. يُحذف الصوت خلال يوم تقريباً والنص خلال ستين يوماً ما لم يُفرض حفظ قانوني. يمكنك الاعتراض عبر المنظّم.

An objection stops the recording: the organiser answers "yes" to `--objections`, the gate refuses
the meeting, nothing is stored, and the refusal is audited with the neutral reason "recording not
used", never naming who objected.

## What the deployer always owns

Whatever the status column says, these stay with the organisation running Praktika:

* the lawful basis, the privacy notice, the consent wording and any regulatory approval or
  notification that applies in your jurisdiction;
* the host: who has root, disk encryption, patching, network egress rules for every process on the
  host, backups and time synchronisation;
* the vault key: generation, storage, access and what happens if it is lost;
* the people: who may run ingests, who reviews, how reviewers are trained, and how an incident is
  handled.
