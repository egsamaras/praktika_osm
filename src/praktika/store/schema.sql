-- Praktika SQLite schema, version 2 (the whole current schema, for a new database).
-- Applied by store.db.migrate(); every statement is idempotent so re-running is safe. An existing
-- database is brought up to date by the upgrade statements in store/db.py instead.
-- Domain objects are stored as canonical JSON next to the columns needed for filtering,
-- retention and search. Times are ISO-8601 strings in UTC.

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id              TEXT PRIMARY KEY,
    title           TEXT    NOT NULL,
    meeting_type    TEXT    NOT NULL,
    classification  TEXT    NOT NULL,
    language_mode   TEXT    NOT NULL,
    platform        TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    organiser       TEXT    NOT NULL,
    private         INTEGER NOT NULL DEFAULT 0,
    legal_hold      INTEGER NOT NULL DEFAULT 0,
    state           TEXT    NOT NULL,
    meeting_json    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_meetings_state ON meetings(state);
CREATE INDEX IF NOT EXISTS ix_meetings_started ON meetings(started_at);

CREATE TABLE IF NOT EXISTS consent (
    meeting_id   TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    record_json  TEXT NOT NULL,
    recorded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    delete_after  TEXT,
    deleted_at    TEXT
);
CREATE INDEX IF NOT EXISTS ix_media_meeting ON media(meeting_id);

CREATE TABLE IF NOT EXISTS transcripts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id     TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    source         TEXT NOT NULL,
    engines_json   TEXT NOT NULL,
    segments_json  TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    redacted       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    delete_after   TEXT,
    deleted_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_transcripts_meeting ON transcripts(meeting_id);

CREATE TABLE IF NOT EXISTS vault (
    meeting_id  TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    blob        BLOB NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS minutes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id       TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    version          INTEGER NOT NULL,
    template         TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    minutes_json     TEXT    NOT NULL,
    provenance_json  TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    UNIQUE (meeting_id, version)
);

CREATE TABLE IF NOT EXISTS review_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id   TEXT    NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    item_id      TEXT    NOT NULL,
    action       TEXT    NOT NULL,
    reason_code  TEXT    NOT NULL,
    before       TEXT,
    after        TEXT,
    by           TEXT    NOT NULL,
    at           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_review_items_meeting ON review_items(meeting_id, version);

-- Append-only mirror of the hash-chained audit log. No foreign key: events such as
-- scope.refused may legitimately reference a meeting that was never saved.
CREATE TABLE IF NOT EXISTS audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    actor           TEXT NOT NULL,
    actor_source    TEXT NOT NULL,
    event           TEXT NOT NULL,
    meeting_id      TEXT,
    classification  TEXT,
    object          TEXT,
    detail_json     TEXT NOT NULL,
    model           TEXT,
    prompt_sha      TEXT,
    prev_hash       TEXT NOT NULL,
    hash            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS holds (
    meeting_id  TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    active      INTEGER NOT NULL,
    reason      TEXT    NOT NULL,
    set_by      TEXT    NOT NULL,
    set_at      TEXT    NOT NULL
);

-- One row per meeting a run is working on (store/locks.py): the command, its process, host and
-- start time. Version 2; the same statement upgrades a version-1 database (store/db.py).
CREATE TABLE IF NOT EXISTS run_locks (
    meeting_id     TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    command        TEXT    NOT NULL,
    pid            INTEGER NOT NULL,
    host           TEXT    NOT NULL,
    started_at     TEXT    NOT NULL,
    token          TEXT    NOT NULL,
    reopened_from  TEXT,
    stopped_by     TEXT
);

-- Microsoft Graph getAllTranscripts/delta cursors per organiser, for a Graph poller that is
-- not built yet (ingest/graph_stub.py).
CREATE TABLE IF NOT EXISTS delta_links (
    organiser_upn  TEXT PRIMARY KEY,
    delta_link     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- Keyword search over minutes. Rows exist only for indexable templates and non-restricted
-- classifications (store/search.py); text is Arabic-normalised before insertion.
CREATE VIRTUAL TABLE IF NOT EXISTS minutes_fts USING fts5(
    meeting_id UNINDEXED,
    version UNINDEXED,
    title,
    summary,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);
