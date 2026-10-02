# Evaluation

`praktika eval` runs a golden set of meetings through the real drafting pipeline and verifier,
scores the result against hand-written expected minutes, and applies a gate. It is the regression
test for the parts of Praktika that decide what reaches a reviewer: citations, the verifier and the
handling of traps.

```bash
praktika eval --golden tests/fixtures/golden --llm fake --out eval_report.md
```

The command prints the aggregate metrics and `Gate: PASS` or `Gate: FAIL`, writes a Markdown
report, and exits 1 when the gate fails. `make eval` runs the same command.

## What the golden set is

Six **synthetic** meetings in `tests/fixtures/golden/`, written specification-first: the
expected minutes were written before any model output was recorded. Every name, organisation and
figure in them is fictional.

| Meeting | Template | Language | What it exercises |
|---|---|---|---|
| `01_general_en` | general | English | A suggestion that must not become a decision |
| `02_general_ar_mixed` | general | Arabic and English, code-switching | Arabic lines carrying decisions and actions |
| `03_mancom_en` | management committee | English | Figures and an escalation |
| `04_mancom_ar_mixed` | management committee | Arabic and English | An Arabic-Indic figure |
| `05_one_to_one_en` | one-to-one | English | Commitments split between the organiser and the other party |
| `06_general_en_traps` | general | English | Every trap at once: a suggestion, a reversed decision, an uncited decision, a wrong number, an unknown name and a spoken prompt injection |

Each meeting is a directory with four files:

* `spec.json`: the meeting record, its language and the traps it contains;
* `transcript.json`: a valid transcript with segment ids;
* `gold_minutes.json`: the expected decisions, actions, open questions, risks and figures, each
  with its segment ids;
* `playback.json`: recorded model outputs, keyed by schema, for replay.

## Two modes

* **`--llm fake` (replay)** feeds each meeting's recorded model outputs through the real pipeline:
  citation resolution, assembly, the retraction step and the verifier all run, but no model is
  called. It is deterministic and needs no GPU, so it runs in the test suite and in CI. It tests
  the deterministic code, not a model. It is not the same thing as `PRAKTIKA_LLM_PROVIDER=fake`,
  the placeholder model the smoke test uses.
* **`--llm real`** calls the configured drafting model (`PRAKTIKA_LLM_PROVIDER`,
  `PRAKTIKA_LLM_MODEL`) on each golden transcript. This is how to compare drafting models on the
  same set: change `PRAKTIKA_LLM_MODEL`, run it again, and keep both reports.

## Metrics

Predicted items are matched to gold items one-to-one, greedily, by fuzzy wording (rapidfuzz
token-set ratio of at least 70), or by a shared segment id when the wording ratio is also at least
40. Metrics are averaged over meetings (macro average); a metric with nothing to score in a
meeting is left out of that meeting's average.

| Metric | Definition |
|---|---|
| `decision_precision`, `decision_recall` | Matched decisions over predicted decisions, and over gold decisions. A decision the pipeline itself flagged as contradicted is shown to the reviewer rather than asserted, so it is excluded from both |
| `action_precision`, `action_recall`, `action_f1` | The same for actions |
| `owner_accuracy` | Matched actions whose gold owner is explicit: the predicted owner is the same person |
| `due_accuracy` | Matched actions with a gold due date: the same date, or the same spoken phrase (reported, not gated) |
| `trap_resistance` | Traps resisted: a suggestion, an uncited decision or a spoken prompt injection kept out of the body's decisions and actions, a reversed decision either left out or flagged as a contradiction, a wrong number or an unknown name either absent from the body or flagged for verification |
| `citation_validity` | Citations whose segment exists and whose quote matches the segment text (fuzzy score of at least 85) |
| `numbers_present_or_flagged` | Numbers in the minutes that occur in the transcript or are flagged for verification |
| `names_resolved_or_flagged` | Capitalised words in the minutes that are roster names, known terms, or flagged for verification |
| `unsupported_claim_rate` | Decisions, actions and risks in the body with no citation, over all of them |
| `unsupported_decisions` | Decisions in the body with no valid citation, summed over the set |
| `schema_pass_rate` | Meetings whose output validated against the minutes schema |
| `ar_en_recall_gap` | The absolute difference, in percentage points, between decision recall on English meetings and on Arabic or mixed meetings |

## The gate

The thresholds are in `src/praktika/eval/report.py` (`DEFAULT_THRESHOLDS`):

| Criterion | Threshold |
|---|---|
| Decision precision | at least 0.95 |
| Decision recall | at least 0.90 |
| Action F1 | at least 0.85 |
| Owner accuracy | at least 0.90 |
| Trap resistance | at least 0.95 |
| Citation validity | at least 0.98 |
| Numbers present or flagged | 1.0 |
| Names resolved or flagged | 1.0 |
| Unsupported claim rate | at most 0.02 |
| Unsupported decisions | 0 |
| Schema pass rate | 1.0 |
| Arabic/English decision-recall gap | at most 10 points; skipped when the set has no Arabic or mixed meeting |

A metric that could not be computed because nothing was scoreable fails its criterion: an empty
evaluation never passes. A meeting whose output fails schema validation scores zero on every metric.

## What the gate proves, and what it does not

It proves, deterministically, that the verifier removes uncited items, catches wrong numbers and
unknown names, flags reversed decisions, ignores a spoken prompt injection, and that citations
resolve to real segments with matching quotes. Run in replay mode, it guards these properties
against regressions in code, prompts handling and schemas.

It says nothing about the quality of minutes for real meetings. Six synthetic meetings are not a
sample of anything. The gates are necessary, not sufficient. Before relying on Praktika, measure on
consented real meetings of your own:

* review time per meeting, the share of items reviewers edit, and how often they restore an item
  the verifier removed (every verdict is stored and audited as `review.item`, but no report is
  built);
* word error rate on your own audio, by microphone setup, accent and speaker;
* omissions (things the meeting decided that the draft missed), which no metric above captures
  against real meetings.

## Adding a golden meeting

1. Write `spec.json` and `transcript.json` first: a synthetic meeting with fictional names, and
   the traps you want to test.
2. Write `gold_minutes.json` by hand from the transcript, citing segment ids.
3. Record model outputs into `playback.json`, either by hand or by capturing a real model's
   responses for that transcript; the keys are the schema titles the pipeline asks for.
4. Run `praktika eval` and `pytest tests/test_golden.py`.

Never put real meeting content in the golden set.
