"""PostgreSQL-backed storage for goal segments and dialogue chunks.

Provides persistent storage of active/archived goal segments so that
goal tracking state survives across pipeline restarts. Falls back to
in-memory storage when the database is unavailable.
"""
import os
import json
from typing import List, Optional, Tuple
from src.config import DialogueChunk

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class GoalStore:
    """Persistent store for goal segments and their chunks.

    Connects to PostgreSQL when DB_HOST is configured in .env,
    otherwise operates as a transparent in-memory fallback so the
    rest of the codebase does not need to care.
    """

    def __init__(self):
        self.conn = None
        self._in_memory_segments: List[dict] = []       # [{segment_id, segment_number, is_active, chunks: [...]}]
        self._in_memory_chunks: List[DialogueChunk] = []
        self._segment_counter: int = 0

        db_host = os.environ.get("DB_HOST", "")
        if PSYCOPG2_AVAILABLE and db_host:
            try:
                self.conn = psycopg2.connect(
                    host=db_host,
                    port=int(os.environ.get("DB_PORT", "5432")),
                    dbname=os.environ.get("DB_NAME", "tierselect"),
                    user=os.environ.get("DB_USER", "tierselect"),
                    password=os.environ.get("DB_PASSWORD", "tierselect123"),
                )
                self.conn.autocommit = True
                self._ensure_tables()
                self._load_segment_counter()
                print(f"[GoalStore] Connected to PostgreSQL at {db_host}:{os.environ.get('DB_PORT', '5432')}")
            except Exception as e:
                print(f"[GoalStore] PostgreSQL connection failed ({e}). Using in-memory fallback.")
                self.conn = None
        else:
            reason = "psycopg2 not installed" if not PSYCOPG2_AVAILABLE else "DB_HOST not set"
            print(f"[GoalStore] {reason}. Using in-memory fallback.")

    @property
    def is_persistent(self) -> bool:
        return self.conn is not None

    # ------------------------------------------------------------------
    # Schema bootstrap (idempotent)
    # ------------------------------------------------------------------
    def _ensure_tables(self):
        """Create tables if they don't exist (matches db_schema.sql)."""
        schema_path = os.path.join(os.path.dirname(__file__), "db_schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                sql = f.read()
            with self.conn.cursor() as cur:
                cur.execute(sql)
        else:
            # Inline minimal schema
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS goal_segments (
                        segment_id VARCHAR(64) PRIMARY KEY,
                        segment_number INT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        archived_at TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS dialogue_chunks (
                        chunk_id VARCHAR(128) PRIMARY KEY,
                        session_id VARCHAR(128) NOT NULL,
                        user_turn TEXT, assistant_turn TEXT, text TEXT,
                        speaker_pair VARCHAR(32) DEFAULT 'user-assistant',
                        prev_chunk_id VARCHAR(128), next_chunk_id VARCHAR(128),
                        timestamp VARCHAR(64),
                        segment_id VARCHAR(64) REFERENCES goal_segments(segment_id) ON DELETE SET NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)

    def _load_segment_counter(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(segment_number), 0) FROM goal_segments")
            self._segment_counter = cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Segment operations
    # ------------------------------------------------------------------
    def create_segment(self, segment_number: int) -> str:
        """Create a new active goal segment. Returns segment_id."""
        segment_id = f"segment_{segment_number}"
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO goal_segments (segment_id, segment_number, is_active) VALUES (%s, %s, TRUE) ON CONFLICT (segment_id) DO NOTHING",
                    (segment_id, segment_number)
                )
        else:
            self._in_memory_segments.append({
                "segment_id": segment_id,
                "segment_number": segment_number,
                "is_active": True,
                "chunks": []
            })
        self._segment_counter = segment_number
        return segment_id

    def archive_segment(self, segment_id: str):
        """Mark a segment as archived and set all its chunks to is_active=False."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE goal_segments SET is_active = FALSE, archived_at = NOW() WHERE segment_id = %s",
                    (segment_id,)
                )
                cur.execute(
                    "UPDATE dialogue_chunks SET is_active = FALSE WHERE segment_id = %s",
                    (segment_id,)
                )
        else:
            for seg in self._in_memory_segments:
                if seg["segment_id"] == segment_id:
                    seg["is_active"] = False
                    for c in seg["chunks"]:
                        c.is_active = False
                    break

    def get_active_segment_id(self) -> Optional[str]:
        """Return the segment_id of the current active segment, or None."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("SELECT segment_id FROM goal_segments WHERE is_active = TRUE ORDER BY segment_number DESC LIMIT 1")
                row = cur.fetchone()
                return row[0] if row else None
        else:
            for seg in reversed(self._in_memory_segments):
                if seg["is_active"]:
                    return seg["segment_id"]
            return None

    def get_segment_counter(self) -> int:
        return self._segment_counter

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------
    def save_chunk(self, chunk: DialogueChunk):
        """Insert or update a dialogue chunk in the store."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dialogue_chunks
                        (chunk_id, session_id, user_turn, assistant_turn, text,
                         speaker_pair, prev_chunk_id, next_chunk_id, timestamp,
                         segment_id, is_active, metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        segment_id = EXCLUDED.segment_id,
                        is_active  = EXCLUDED.is_active,
                        metadata   = EXCLUDED.metadata
                """, (
                    chunk.chunk_id, chunk.session_id, chunk.user_turn,
                    chunk.assistant_turn, chunk.text, chunk.speaker_pair,
                    chunk.prev_chunk_id, chunk.next_chunk_id, chunk.timestamp,
                    chunk.segment_id, chunk.is_active,
                    json.dumps(chunk.metadata)
                ))
        else:
            # In-memory: add to the matching segment's chunk list
            existing = [c for c in self._in_memory_chunks if c.chunk_id == chunk.chunk_id]
            if existing:
                existing[0].is_active = chunk.is_active
                existing[0].segment_id = chunk.segment_id
            else:
                self._in_memory_chunks.append(chunk)
                for seg in self._in_memory_segments:
                    if seg["segment_id"] == chunk.segment_id:
                        seg["chunks"].append(chunk)

    def get_active_chunks(self) -> List[DialogueChunk]:
        """Return all chunks belonging to the current active segment."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, session_id, user_turn, assistant_turn, text,
                           speaker_pair, prev_chunk_id, next_chunk_id, timestamp,
                           segment_id, is_active, metadata
                    FROM dialogue_chunks
                    WHERE is_active = TRUE
                    ORDER BY created_at
                """)
                return [self._row_to_chunk(row) for row in cur.fetchall()]
        else:
            return [c for c in self._in_memory_chunks if c.is_active]

    def get_archived_chunks(self) -> List[DialogueChunk]:
        """Return all chunks belonging to archived segments."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, session_id, user_turn, assistant_turn, text,
                           speaker_pair, prev_chunk_id, next_chunk_id, timestamp,
                           segment_id, is_active, metadata
                    FROM dialogue_chunks
                    WHERE is_active = FALSE
                    ORDER BY created_at
                """)
                return [self._row_to_chunk(row) for row in cur.fetchall()]
        else:
            return [c for c in self._in_memory_chunks if not c.is_active]

    def get_all_chunks(self) -> List[DialogueChunk]:
        """Return all stored chunks (active + archived)."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, session_id, user_turn, assistant_turn, text,
                           speaker_pair, prev_chunk_id, next_chunk_id, timestamp,
                           segment_id, is_active, metadata
                    FROM dialogue_chunks ORDER BY created_at
                """)
                return [self._row_to_chunk(row) for row in cur.fetchall()]
        else:
            return list(self._in_memory_chunks)

    def get_archived_segment_chunks(self) -> List[List[DialogueChunk]]:
        """Return archived segments as list of chunk-lists (matching old GoalTracker.archived_segments)."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT gs.segment_id
                    FROM goal_segments gs
                    WHERE gs.is_active = FALSE
                    ORDER BY gs.segment_number
                """)
                segments = []
                for (seg_id,) in cur.fetchall():
                    cur.execute("""
                        SELECT chunk_id, session_id, user_turn, assistant_turn, text,
                               speaker_pair, prev_chunk_id, next_chunk_id, timestamp,
                               segment_id, is_active, metadata
                        FROM dialogue_chunks WHERE segment_id = %s ORDER BY created_at
                    """, (seg_id,))
                    segments.append([self._row_to_chunk(row) for row in cur.fetchall()])
                return segments
        else:
            return [seg["chunks"] for seg in self._in_memory_segments if not seg["is_active"]]

    # ------------------------------------------------------------------
    # Reset / cleanup
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all segments and chunks (for fresh experiment runs)."""
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM dialogue_chunks")
                cur.execute("DELETE FROM goal_segments")
        self._in_memory_segments = []
        self._in_memory_chunks = []
        self._segment_counter = 0

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_chunk(row) -> DialogueChunk:
        meta = row[11] if row[11] else {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return DialogueChunk(
            chunk_id=row[0],
            session_id=row[1],
            user_turn=row[2] or "",
            assistant_turn=row[3] or "",
            text=row[4] or "",
            speaker_pair=row[5] or "user-assistant",
            prev_chunk_id=row[6],
            next_chunk_id=row[7],
            timestamp=row[8],
            segment_id=row[9],
            is_active=row[10] if row[10] is not None else True,
            metadata=meta
        )

