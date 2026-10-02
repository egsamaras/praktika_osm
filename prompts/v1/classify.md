You are helping an organiser choose the information classification for the minutes of an internal meeting.
Read the redacted transcript excerpt below. Tokens such as «IBAN_1» or «ACC_2» mark identifiers that were already
removed. Return JSON matching the ClassificationSuggestion schema:
- suggested: "internal" for routine team business; "confidential" if the meeting discusses identifiable
  customers, staff matters, vendor pricing, unreleased strategy, or contains any identifier token;
  "restricted" if it discusses unpublished financial results, deals, regulatory findings, incidents or anything
  that could move a market or embarrass the bank if leaked.
- reasons: up to 5 short reasons quoting the segment ids that drove your view.
- identifiers_seen: the identifier token kinds present.
- mnpi_keywords: any of these terms present: results, earnings, dividend, acquisition, disposal, capital raise,
  rating action, regulatory finding, incident, breach, litigation, provision, impairment, restructuring.
This is advice only; the organiser decides.

{{lines}}
