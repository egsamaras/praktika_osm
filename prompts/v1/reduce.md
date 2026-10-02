You are merging findings extracted from consecutive excerpts of one meeting. You do not have the transcript; you
have only the findings below and the roster. Produce JSON matching the MergedFindings schema:
- Merge duplicates (same decision or action phrased differently, or sharing segment ids), keeping the union of
  their segment ids and the clearer wording.
- If two findings conflict (a decision approved then deferred; two owners for one action), keep the later one by
  segment order, record the earlier one in retracted_decisions where applicable, and keep both sets of ids.
- Order everything by first segment id. Keep quotes exactly as given; never write new quotes.
- Do not add any item that is not already present in the findings. Do not add segment ids that are not present.

Findings:
{{findings_json}}
