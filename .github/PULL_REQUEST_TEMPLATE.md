### What this changes, and why

<!-- A short description. Link the issue it fixes, if there is one. -->

### Checklist

- [ ] Synthetic data only: no real meeting content, recordings, transcripts, names or identifiers
      in code, tests, fixtures, the golden set, logs or this description.
- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check src tests` and `uv run ruff format --check src tests` pass.
- [ ] `uv run praktika eval --llm fake` ends with `Gate: PASS`.
- [ ] Tests added or updated for every change in behaviour.
- [ ] No way added to skip the consent gate, reach a host outside the allow-list, export
      unapproved minutes, send anything anywhere, or compute per-person analytics.
- [ ] If `prompts/` changed: `PINNED_SHA256` in `src/praktika/llm/prompts.py` is updated and the
      reason is given above.
- [ ] If a dependency changed: `uv.lock` is updated with `uv lock` in this pull request.
- [ ] A line is added to `CHANGELOG.md` under "Unreleased".
- [ ] Documentation is updated, in British English.

By opening this pull request, I agree that my contribution is licensed under the Apache License
2.0, as the rest of the project is.
