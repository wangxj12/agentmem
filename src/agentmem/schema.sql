-- AgentMemory 0.1.0; database schema 6: Base and Fact memory.
CREATE TABLE IF NOT EXISTS add_requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (request_id, user_id, session_id)
);

CREATE TABLE IF NOT EXISTS raw_messages (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    speaker TEXT,
    source_timestamp INTEGER,
    fact_context_seq INTEGER UNIQUE,
    UNIQUE(id,user_id),
    UNIQUE(request_id,message_index),
    FOREIGN KEY(request_id,user_id,session_id) REFERENCES add_requests(request_id,user_id,session_id)
);
CREATE INDEX IF NOT EXISTS raw_messages_context_idx ON raw_messages(user_id,fact_context_seq);

CREATE TABLE IF NOT EXISTS raw_chunks (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source_timestamp INTEGER,
    speaker TEXT,
    created_at TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    FOREIGN KEY (message_id,user_id) REFERENCES raw_messages(id,user_id),
    FOREIGN KEY (request_id, user_id, session_id)
        REFERENCES add_requests(request_id, user_id, session_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS raw_chunks_identity_idx ON raw_chunks(id,user_id);
CREATE INDEX IF NOT EXISTS raw_chunks_message_idx ON raw_chunks(message_id);
CREATE INDEX IF NOT EXISTS raw_chunks_user_idx ON raw_chunks(user_id);

-- Mem0 vector payload and vector share a single SQLite record.
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    text_lemmatized TEXT NOT NULL,
    linked_memory_ids TEXT NOT NULL,
    attributed_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    FOREIGN KEY (request_id, user_id, session_id)
        REFERENCES add_requests(request_id, user_id, session_id)
);
CREATE INDEX IF NOT EXISTS facts_user_idx ON facts(user_id);

CREATE TABLE IF NOT EXISTS fact_entities (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    entity_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    linked_fact_ids TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_entities_user_idx ON fact_entities(user_id);

CREATE TABLE IF NOT EXISTS store_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Fact provenance table; no extra history, score or retrieval tables.
CREATE UNIQUE INDEX IF NOT EXISTS facts_identity_idx ON facts(id,user_id);
CREATE TABLE IF NOT EXISTS memory_sources (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_role TEXT NOT NULL CHECK(source_role='fact'),
    source_start INTEGER NOT NULL CHECK(source_start >= 0),
    source_end INTEGER NOT NULL CHECK(source_end > source_start),
    FOREIGN KEY(fact_id,user_id) REFERENCES facts(id,user_id),
    FOREIGN KEY(chunk_id,user_id) REFERENCES raw_chunks(id,user_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS fact_source_unique ON memory_sources(fact_id,chunk_id,source_role,source_start,source_end) WHERE fact_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sources_user_idx ON memory_sources(user_id);

-- Operational status only: original chunk/fact tables and write logic are unchanged.
-- No foreign key to add_requests: this row must also describe raw write failures.
CREATE TABLE IF NOT EXISTS request_states (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    processing INTEGER NOT NULL DEFAULT 0 CHECK(processing IN (0,1)),
    chunk_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(chunk_status IN ('pending','running','succeeded','failed')),
    fact_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(fact_status IN ('pending','running','succeeded','failed','disabled')),
    chunk_attempts INTEGER NOT NULL DEFAULT 0,
    fact_attempts INTEGER NOT NULL DEFAULT 0,
    chunk_error TEXT,
    fact_error TEXT,
    fact_retryable INTEGER NOT NULL DEFAULT 1,
    fact_warnings TEXT NOT NULL DEFAULT '[]',
    fact_expected_hashes TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS request_states_user_idx ON request_states(user_id);
