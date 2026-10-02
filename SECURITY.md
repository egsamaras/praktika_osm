# Security policy

Praktika handles meeting content, so security reports are welcome and taken seriously.

## Reporting a vulnerability

Report vulnerabilities **privately**, through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability** (this creates a draft security
advisory that only the maintainers can see). Please do not open a public issue, pull request or
discussion for a suspected vulnerability.

If the **Report a vulnerability** button is missing, private reporting is not switched on yet. In
that case, open an ordinary issue that asks for a private contact and says nothing else: no
details of the vulnerability, the affected component or how to reproduce it. A maintainer will
reply with a private channel, and the report continues there.

Include:

* the version or commit you tested;
* the platform (macOS or Linux, local or service mode) and any non-default settings;
* the steps to reproduce, and what an attacker gains;
* a proof of concept if you have one.

**Never include real meeting content, recordings, transcripts, personal data or credentials.**
Reproduce with the synthetic fixtures in `tests/fixtures/` or with data you have made up.

This is a small open-source project maintained on a best-effort basis. You will get an
acknowledgement, and a fix or a published advisory where one is warranted; please allow
reasonable time before any disclosure, and say if you would like to be credited.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| anything older | No |

## Scope

In scope: the code in this repository, in particular anything that defeats a control described in
[docs/CONTROLS.md](docs/CONTROLS.md). For example:

* reaching a host outside `PRAKTIKA_ALLOWED_HOSTS`, or making model calls with untokenised
  identifiers;
* storing a meeting without passing the consent and scope gate, or skipping the gate;
* exporting, approving or de-tokenising minutes that policy should refuse;
* bypassing the review server's session token, `Host`/`Origin` checks, CSRF header, OIDC token
  validation or role checks;
* forging or truncating the audit chain without `praktika audit verify` noticing, by anyone who is
  not root on the host;
* retention or legal-hold logic that keeps data past its timer or deletes data under hold;
* path traversal, injection or unsafe file permissions in the CLI, the review server or the
  exports;
* prompt-injection content in a transcript that gets an item past the verifier into approved
  minutes without a flag.

Out of scope (report these upstream or treat them as known limitations):

* vulnerabilities in third-party components (Ollama, vLLM, Whisper, PyTorch, other dependencies) —
  please report them to their maintainers; tell us if Praktika's use of them makes things worse;
* attacks that need root or the service account on the host, which can read everything by design;
* deployments that switch controls off: `PRAKTIKA_PILOT_SMOKE=true` outside a smoke test, a
  wildcard-style allow-list, a review server exposed without the documented gateway, or model
  servers bound to a non-loopback address;
* the limitations stated in [docs/LIMITATIONS.md](docs/LIMITATIONS.md), unless you show a way
  around the mitigation described there;
* the quality of drafted minutes (a wrong or missing item that the verifier and the reviewer are
  meant to catch); please open an ordinary issue with synthetic data instead.
