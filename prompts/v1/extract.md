Read the transcript excerpt below (chunk {{index}} of {{total}}) and extract, as JSON matching the ChunkFindings
schema, only what this excerpt directly supports:
- decisions: concluded outcomes with kind (approved | noted | deferred | rejected | agreed_in_principle), who
  decided (a person from the roster, "Chair" or "Committee"), any dissent or conditions, the citing segment ids
  and one verbatim quote (max 240 characters, copied exactly from one cited segment).
- actions: things a person or group committed to do ("I will ...", "سأقوم ...", "Omar to ..."), one per
  commitment. Fill owner_evidence first: copy the words that show who owns it (the speaker label if it is a
  name, or a line elsewhere that names them, such as the chair's summary "Omar drafts the notice by the 18th"
  or "thanks Omar" following the commitment); then owner in roster spelling (owner_confidence "explicit" when
  the speaker label is the name, "inferred" when it came from another line, "unknown" with owner "" only when
  nothing names them); due_text copied as spoken (never empty when a time was stated); citing ids and a
  verbatim quote. Commitments spoken in Arabic count exactly like English ones.
- questions: open questions explicitly left unanswered, with who raised them.
- risks: risks or concerns explicitly raised, with severity as stated or "medium" if not stated.
- key_points: 1 to 6 short points per topic discussed, each with citing ids.
- figures: every number, amount or percentage spoken with its context and citing id.
Lines tagged |ar or |mixed are Arabic: read each of them separately and check it for a commitment (سأ / سوف /
راح / أنا أتولى ...), a decision (قررنا / نوافق / اتفقنا / خلاص ...) or a question before you finish; report them
in English with the Arabic quote copied verbatim. For an unlabelled speaker, look through the whole excerpt for
a line that names who made the commitment (a chair's summary, "thanks Omar", a reply) before leaving owner empty.
Omit anything that is only suggested, hypothetical or later withdrawn within this excerpt. Empty lists are fine.
Optional text fields (owner, due_text, raised_by, mitigation, dissent_or_conditions) are strings: write "" when
the transcript genuinely does not state them, never a placeholder.

Transcript excerpt:
{{lines}}
