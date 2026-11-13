-- TierSelect PostgreSQL Schema
-- Auto-executed on first docker-compose up via /docker-entrypoint-initdb.d/

-- Goal segments (active and archived)
CREATE TABLE IF NOT EXISTS goal_segments (
    segment_id   VARCHAR(64) PRIMARY KEY,
    segment_number INT NOT NULL,
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMP DEFAULT NOW(),
    archived_at  TIMESTAMP
);

-- Dialogue chunks with goal segment linkage
CREATE TABLE IF NOT EXISTS dialogue_chunks (
    chunk_id       VARCHAR(128) PRIMARY KEY,
    session_id     VARCHAR(128) NOT NULL,
    user_turn      TEXT,
    assistant_turn TEXT,
    text           TEXT,
    speaker_pair   VARCHAR(32) DEFAULT 'user-assistant',
    prev_chunk_id  VARCHAR(128),
    next_chunk_id  VARCHAR(128),
    timestamp      VARCHAR(64),
    segment_id     VARCHAR(64) REFERENCES goal_segments(segment_id) ON DELETE SET NULL,
    is_active      BOOLEAN DEFAULT TRUE,
    metadata       JSONB DEFAULT '{}',
    created_at     TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups by segment and active status
CREATE INDEX IF NOT EXISTS idx_chunks_segment ON dialogue_chunks(segment_id);
CREATE INDEX IF NOT EXISTS idx_chunks_active  ON dialogue_chunks(is_active);
CREATE INDEX IF NOT EXISTS idx_segments_active ON goal_segments(is_active);

