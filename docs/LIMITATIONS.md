# Limitations

This page lists what Praktika cannot yet do well, what it deliberately does not do, and what is
not built. Read it before you put real meetings through the tool. Release 0.1.0 is pilot-grade:
the controls are in code and tested, but the tool has not been validated on real meetings at any
scale.

## The review is the product, and it costs time

* Every draft needs a named person to check it against the transcript, resolve its flags, name
  unmapped speakers and approve it. Nothing can be exported before that. The review page exists
  because the draft is not trustworthy on its own: **names, figures, dates, owners and decisions
  must be confirmed by a human.**
* How long a review takes on a real meeting has not been measured.
* A reviewer can approve without reading. Flags are typed and prioritised, and only priority-1
  flags block approval, to keep the load down; but anyone who may edit a meeting can clear any flag
  or restore a removed item with one click, and neither action re-runs the verifier. Both are
  audited, not prevented.
* The draft is not immediate. Drafting makes one extraction call per chunk of transcript, a merge
  call when there are several chunks, a narrative call and one retraction-check call per decision.
  The sizing estimate for a 60-minute meeting on a single data-centre GPU is 10 to 15 minutes of
  machine time, speech-to-text included; it has not been measured. On a laptop it is much slower:
  during development, on an Apple M4 with 24 GB of memory, `qwen2.5:14b` generated about 4 tokens
  a second, a two-minute meeting took three to four minutes to draft, and a 22-minute synthetic
  Teams transcript about 8 minutes. With 16 GB of memory the default model does not fit in what
  macOS lets the GPU use, and drafting is slower still.

## The verifier catches some errors, not all

* The verifier is deterministic. It checks that every cited segment exists, that each quote
  matches its segment (fuzzy score of at least 85, with the same negations and the same digits),
  that numbers in the minutes occur in the transcript and that names are on the roster or in the
  glossary. Decisions and actions left without a valid citation are moved into priority-1 flags
  that carry their full text.
* It cannot see an error the speech engine made consistently: a misheard figure is "present in the
  transcript". It checks that a quote is there, not that the item draws the right conclusion from
  it.
* Its prompt-injection guard is a small, deliberately narrow list of instruction-like phrases
  aimed at the model. An injection that reads like ordinary meeting speech will not match. The real
  control is the citation requirement plus human approval.
* The verifier writes no audit event: an item it removes is visible in the stored minutes, not in
  the audit stream.
* The "Figures mentioned" list of management-committee (`mancom`) minutes is not verified at all.
  It is copied from the model's findings: its segment ids are not resolved against the
  transcript, so an id that does not exist is kept, and its numbers are not checked against the
  transcript. The review page does not show the list and pilot mode refuses draft exports, so a
  reviewer first sees it in the Markdown or DOCX export, after approval. Check every figure there.
* Due dates are worked out from the spoken phrase ("by Friday next week", "the 15th of next
  month") by fixed rules, and some phrasings still come out as a confident but wrong date. The
  verifier does not check them. The review page and `praktika actions` show the spoken phrase next
  to each date so that the reviewer can; exports show the date only.

## Identifier tokenisation

* Tokenisation is pattern-based. It covers e-mail addresses, IBANs, account and card numbers,
  Bahraini CPR and Saudi iqama or national ID numbers, phone numbers in Bahraini, Saudi and UK
  formats, and amounts spoken near a person's name. Phone numbers from other countries, including
  the other Gulf states, and other national identifiers are not tokenised, and neither is an
  identifier the speech engine wrote out in words. Names are not tokenised.

## Speech recognition

* Two speech paths exist: Whisper large-v3-turbo in process on Apple silicon (MLX), and Whisper
  large-v3-turbo served by vLLM over HTTP on a Linux GPU server. The MLX path was exercised during
  development on synthetic recordings and on short single-speaker English microphone tests. The
  HTTP backend has been tested only against a mocked transport in the test suite.
* No word error rate has been measured on real meeting audio. Built-in laptop microphones and
  meeting-room microphones will do worse than any published benchmark.
* Word timestamps depend on the speech server returning them. If it does not, Praktika silently
  falls back to chunk-level timing and every citation becomes up to 28 seconds wide instead of
  word-precise. [DEPLOYMENT.md](DEPLOYMENT.md) has a probe for this.
* A Teams transcript is Microsoft's transcription, with Microsoft's errors. Praktika drafts from it
  as it stands. The glossary's misrendering fixes are applied to Praktika's own speech-to-text
  output only, not to an imported transcript.

## Arabic

* The Arabic speech path is off by default (`PRAKTIKA_STT_AR=none`): every chunk is decoded as
  English and `--lang ar-mixed` is refused before the consent script is shown. When it is switched
  on for development, chunks are routed by Whisper's language identification to an Arabic engine
  (Cohere Transcribe Arabic on MLX, or Whisper large-v3). It has been exercised only on synthetic
  audio.
* On the HTTP speech path there is no language detector, so `--lang auto` resolves to English.
* Minutes are always written in English; Arabic quotes are kept verbatim with an English gloss.
  There is no Arabic minutes template.

## The drafting model

* `qwen2.5:14b` on Ollama is the configuration exercised so far. The OpenAI-compatible client
  (for vLLM) has been tested against a mocked server only. A dense 32-billion-parameter model is
  likely to be too slow on bandwidth-limited unified-memory hardware; measure before choosing one.
* With Ollama, `PRAKTIKA_LLM_NUM_CTX` must stay at 32768: with a smaller context Ollama truncates
  the transcript, and Praktika only logs a warning (`ollama.prompt_truncated`); the run does not
  stop. Praktika sends the value on every call. At that context `qwen2.5:14b` takes about 15 GB
  while loaded, so a Mac needs 24 GB of memory to run it comfortably; on 16 GB, a smaller model
  such as `qwen2.5:7b` fits but has not been evaluated (see the
  [README](../README.md#on-a-16-gb-mac)).
* The long-transcript fallback model is not checked by `praktika doctor`. Its default is
  `llama3.1:8b`; if that model is not present, a long transcript fails. On vLLM, which serves one
  model per server, any fallback name other than the served model produces an HTTP error rather
  than a degraded draft. Set `PRAKTIKA_LLM_FALLBACK_MODEL` to the same value as
  `PRAKTIKA_LLM_MODEL`.
* "Regenerate" on the review page sends the whole transcript in one call with no chunking, so on a
  long meeting it can exceed the context window.
* `PRAKTIKA_LLM_PROVIDER=fake` (the smoke test) is a placeholder, not a summariser: it quotes the
  transcript verbatim so that the pipeline can be exercised without a model, its "decisions" carry
  no judgement, and every draft it produces is marked as fake.

## Speaker attribution

* Diarisation is off by default. When enabled it produces anonymous labels (`SPEAKER_01`) and
  times only, never matched across meetings. Enabling it is a deployer decision that should follow
  a data-protection assessment of the transient speaker embeddings the model computes.
* Teams transcripts carry display names; recordings do not. The reviewer maps labels to names on
  the review page, and an unmapped label in a citation raises an `unresolved_speaker` flag. For a
  meeting-room recording, attribution therefore depends entirely on the reviewer, and so do action
  owners whenever speakers are unlabelled.
* A room microphone records whoever is in the room; in a hybrid meeting it hears the far end only
  through the room loudspeaker.
* One-to-one minutes put a commitment under "my commitments" only when it cites the organiser's own
  microphone track, which exists only in live capture. Transcript files and recordings have no such
  track, so every commitment lands under "their commitments" and the reviewer has to read the
  owners.

## Scope and consent

* Pilot mode (`PRAKTIKA_PILOT=true`, the default) accepts Internal meetings only and refuses board,
  board sub-committee, HR, customer, regulator or auditor, legally privileged and foreign-hosted
  meetings in code when the organiser tags them so. Outside pilot mode, `confidential` and
  `restricted` meetings need `PRAKTIKA_MODE=service`. The classification comes from `--class`
  and defaults to `internal` without being asked for, and the exclusion tags are the organiser's
  own answers: a mis-classified or untagged meeting will be processed.
* The scope checklist is required in every mode, not only in pilot mode: the gate refuses a
  meeting unless every checklist item (not a board meeting, not HR, and so on) is confirmed. So
  outside pilot mode a board meeting is still refused unless the organiser confirms, untruthfully,
  that it is not one; only the tag-based exclusions are switched off.
* `--external-participants` records the tag `external` on the meeting; it does not refuse the
  meeting. Whether guests may be recorded at all is a deployer decision.
* Consent is **procedural**: the organiser attests that attendees were notified and nobody
  objected, and the record is audited, but nothing verifies the attestation. The gate has no skip
  flag, and the test suite checks that none appears; it can still be answered untruthfully. That is
  a people control, not a code control.
* **What is kept, and for how long.** This is what the code does, and what the consent wording
  and your privacy notice have to describe:
  * Deletion happens at a retention run, never at the moment of approval: at the start of every
    command that opens the meeting store (`ingest`, `start`, `transcribe`, `generate`, `abort`,
    `approve`, `export`, `search`, `actions`, `hold`, `dsar`), once when `praktika serve` starts
    (not again while it runs, so the audio of a meeting approved on the review page stays until
    the next run), with `praktika retention run`, and hourly from the scheduler that
    `praktika retention install` sets up. Without the scheduler every deadline below can slip for
    as long as nobody runs a command.
  * Converted audio is kept for click-to-play review. It is deleted at the first run after the
    minutes are approved or discarded, and in any case at the first run once
    `PRAKTIKA_RETENTION_AUDIO_HOURS` have passed since conversion: 24 hours for Internal meetings
    and 72 for Confidential by default. Where the value is 0 (Restricted, by default) it is
    deleted at the first run after transcription, and the review page has no audio playback.
  * The transcript and its token vault are deleted `PRAKTIKA_RETENTION_TRANSCRIPT_DAYS` after
    approval (14 days by default, 7 for Restricted), and in any case 60 days after the transcript
    was first stored (30 for Restricted), whether or not the minutes are ever approved. The 60-
    and 30-day maximum is fixed in code (`retention.TRANSCRIPT_MAX_DAYS`); no setting changes it.
  * Unapproved draft versions are deleted `PRAKTIKA_RETENTION_DRAFT_DAYS` after approval or, for a
    meeting that is never approved, after its latest draft was made (30 days by default, 14 for
    Restricted), whether the meeting was discarded or simply left.
  * Approved minutes have no timer; only a data-subject deletion removes them. Exports and the
    source files you ingested are never deleted by Praktika (see below). A legal hold stops every
    timer for its meeting.
* **Who can read a draft.** In local mode, anyone using the review page acts as the account that
  started `praktika serve`. In service mode, besides the organiser, the secretary role can read,
  edit, approve and discard the minutes of every meeting that is not private (one-to-ones are
  private), and the DPO and administrator roles can read every meeting, including drafts,
  transcripts and retained audio.
* The spoken consent script and the chat notice, and the settings they were written for, are in
  [CONTROLS.md](CONTROLS.md#consent-script-and-chat-notice). A deployment that differs (other
  retention settings, no scheduler, service mode) must change the wording, and the privacy notice
  must describe what the deployment actually does.
* `praktika ingest` never deletes the file you give it; remove the source file yourself once its
  transcript is stored.

## Identity, until single sign-on is in place

* In local mode the identity is the operating-system account: on Linux the account the user
  logged in with (the kernel login uid, kept across `sudo -u`), recorded as `source=local`; on
  macOS the console user. That is not a directory-verified identity.
* The review page acts as whoever started `praktika serve`. Anyone using that page can read, edit,
  approve and export every meeting that account organises, and every page action is audited
  against that one name. Any account that can log in to the host and read the data directory has
  full API access as the local operator.
* The local session token is persistent per data directory and never expires; there is no logout
  and no idle timeout. Deleting the token file and restarting `serve` issues a new one.
* Service mode (OIDC against an issuer's JWKS, directory-group or app-role roles, a non-loopback
  bind behind a gateway) is implemented and tested with synthetic tokens only; it has not run
  against a real identity provider. The static review page sends no `Authorization` header, so a
  browser sign-in needs an authenticating gateway that injects the bearer token, or new front-end
  code. The identity provider must emit the group **names** (`Praktika-Users`,
  `Praktika-Secretaries`, `Praktika-DPO`, `Praktika-Admins`) or app roles, not group object IDs, or
  every user is refused.
* The command line has no identity under OIDC, so an ingest worker runs with the `session`
  provider and the consent record names the operator. `--organiser` names the real organiser;
  nothing checks that name against a directory.

## Storage and operations

* One SQLite file guarded by a process-level lock: fine for one reviewer, not for a shared service.
  A PostgreSQL store and object storage are not built (the `Store` protocol is the seam). The data
  directory must be on a local filesystem, because the audit lock is an `flock`.
* Deleting a transcript, vault or draft is a database operation. Praktika does not set
  `PRAGMA secure_delete`, so freed pages may persist until SQLite reuses them. Put the data
  directory on an encrypted volume. Deleted transcripts keep a tombstone row (hash and timestamps);
  approved minutes are removed only by a DSAR deletion.
* Exported files in `<data_dir>/exports` are not under any retention timer.
* Backups: a backup of the data directory outlives every retention timer. Exclude the data
  directory from backup, or make the backup inherit the retention clock.
* A first ingest that fails leaves the meeting at `created`, with its consent record, and the
  review page lists it. If the audio could not be transcribed (for example with the speech server
  down), the run purges it and audits `ingest.failed`. If the audio has no speech or the transcript
  file is empty, a registered WAV stays under its retention timer so that
  `praktika transcribe <meeting-id>` can retry. If drafting fails, the transcript is kept and
  `praktika generate <meeting-id>` drafts again. `praktika abort <meeting-id>` discards the meeting
  and removes what is left.
* There is no systemd unit for `praktika serve`: the review page does not survive a reboot, and the
  vault key, an exported environment variable on Linux, must be set again after every reboot. The
  hourly retention timer is a systemd unit (a launchd agent on macOS) and does survive.
* The vault key has no escrow, no rotation path and no re-keying command. Losing it makes
  de-tokenisation of earlier meetings impossible (the tokenised minutes stay readable); leaking it
  re-identifies every token in every transcript still on the host.
* Root on the host can read every transcript, the retained audio, the exports and the vault key,
  and can rewrite the local audit chain, its head file and the database consistently. The defences
  are short retention, a short list of administrators, and shipping the audit stream off the host.
* The forwarded copy of the audit stream (`audit-forward.jsonl`, with `PRAKTIKA_AUDIT_SINK=stdout`)
  is written, but nothing in this repository ships it; `doctor` only checks for a marker file that
  a log-shipping agent is expected to write. On the review server, authentication refusals (401)
  and authorisation refusals (403 for a meeting the identity may not read or change) are audited
  as `auth.denied`. Refusals by the request guard (wrong `Host`, missing CSRF header or session
  token) and other refused requests (for example 404, 409, or an approval refused while a
  priority-1 flag is open) do not reach the audit chain; at most they appear in the application
  log.
* The vLLM speech server and Ollama accept any local caller without authentication, and
  Praktika's speech client has no credential field. Bind both to loopback.
* With the speech backend on HTTP, the registered model hashes describe the weights directory you
  registered, not what the speech server actually loaded.
* One host processes one meeting at a time; there is no job queue or scheduler.

## Evaluation

* The golden set is six **synthetic** meetings written specification-first. It proves the
  verifier, citations and trap handling deterministically and says nothing about quality on real
  meetings. The gates in `praktika eval` are necessary, not sufficient. See
  [EVALUATION.md](EVALUATION.md).
* There is no LLM-as-judge, no reviewer-concordance data and no word error rate on real audio.

## Platform

* Supported hosts: Apple silicon Macs (development and single-user use, speech in process on MLX)
  and Linux on x86_64 or aarch64 with an NVIDIA GPU (speech through vLLM over HTTP, drafting through
  Ollama). Python 3.12 only. There is no Windows support.
* On Linux, the `cuda` extra (faster-whisper) is a development backend. On aarch64 it installs and
  then runs on the CPU only, about thirty times too slow, without an error; do not install it
  there.
* There is no maintained container image of Praktika itself. `docker/linux-test.Dockerfile` builds
  an image that runs the test suite on Linux; it is not a deployment image.
* `praktika start --source sck` expects a signed macOS system-audio capture helper that is not part
  of this repository; only microphone capture (`--source mic`) works out of the box.

## What is not built

* Browser upload with the consent gate rendered as a web form; today meetings are submitted from
  the command line.
* Browser sign-in for the review page, other than through an authenticating gateway.
* Integration testing of OIDC against a real identity provider and gateway.
* A Microsoft Graph transcript poller: only the protocol and its error vocabulary exist
  (`ingest/graph_stub.py`).
* Shipping of the audit stream to a SIEM, with parsers and detections.
* A PostgreSQL store and object storage.
* A vLLM service profile for drafting as well as speech, tested end to end.
* Chunking in "regenerate section".
* The speech server's weights digest carried into provenance.
* Request-guard refusals (`Host`, CSRF header, session token) and refused requests other than
  authentication and authorisation refusals through the audit chain.
* Verification of the management-committee "Figures mentioned" list, and its display on the
  review page.
* A credential for the speech endpoint.
* `PRAGMA secure_delete=ON` and a periodic `VACUUM`.
* Exports under a retention timer.
* An audit event from the verifier, for prompt-injection detection.
* A length bound on speaker names mapped on the review page.
* A systemd unit for `praktika serve`.
* Minutes templates beyond `general`, `mancom` (management committee) and `one_to_one`.
* Live capture of conferencing-platform audio on a server.

## Out of scope (no code path exists)

Sending e-mail or chat messages, creating calendar events or tasks, meeting bots, cloud speech,
LLM or embedding APIs, voiceprints or cross-meeting speaker identification, per-person analytics
(talk time, sentiment, attendance scoring), audio export, live captions, fine-tuning and semantic
search. Adding any of these changes what the tool is; treat it as a material change that needs a
fresh assessment in your organisation.
