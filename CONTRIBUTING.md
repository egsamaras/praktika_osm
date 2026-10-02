# Contributing

Thank you for considering a contribution. Issues and pull requests are welcome.

## Ground rules

* **Synthetic data only.** Never put real meeting content, recordings, transcripts, names of real
  people in a meeting context, personal data or credentials in an issue, a pull request, a test or
  a fixture. Make up a meeting, or use the fixtures in `tests/fixtures/`. When you report a bad
  draft, reproduce it on synthetic data first.
* **Security issues go through [SECURITY.md](SECURITY.md)**, not the public tracker.
* **Be respectful.** The project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
* **Keep the controls intact.** A change must not add a way to skip the consent gate, call a host
  outside the allow-list, export unapproved minutes, send anything anywhere, or compute per-person
  analytics. Tests guard several of these; if one fails, the change is wrong, not the test.

## Before you open a pull request

On Apple silicon, add `--extra mac` to the first line.

```bash
uv sync --frozen --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv run praktika eval --llm fake
```

The evaluation must end with `Gate: PASS`. The test suite runs offline and needs no model weights
or GPU; CI runs the same checks on Linux. [docs/DEVELOPING.md](docs/DEVELOPING.md) describes the
setup and the conventions; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#extending-it) covers adding
a backend or a minutes template, and
[docs/EVALUATION.md](docs/EVALUATION.md#adding-a-golden-meeting) adding a golden meeting.

* Add or update tests with every behaviour change.
* If you change anything under `prompts/`, update the pinned hash in
  `src/praktika/llm/prompts.py` and say why in the pull request.
* If you change a dependency, update `uv.lock` with `uv lock` in the same pull request.
* Add a line to `CHANGELOG.md` under "Unreleased".
* Write documentation in British English.

## Licence

By contributing, you agree that your contribution is licensed under the Apache License 2.0, as
the rest of the project is (see [LICENSE](LICENSE)).
