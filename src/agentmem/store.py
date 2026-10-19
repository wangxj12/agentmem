"""SQLite persistence and transaction boundaries for memory data."""

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

from .requests import MemoryMessage, RequestConflictError, message_id, payload_hash

SCHEMA_VERSION = "6"
APPLICATION_TABLES = frozenset(
    {
        "add_requests",
        "raw_chunks",
        "facts",
        "fact_entities",
        "store_metadata",
        "raw_messages",
        "memory_sources",
        "request_states",
    }
)
BUSY_TIMEOUT_MS = 60_000  # SQLite writer lock wait limit, aligned with vanilla memory_store._connect.
EMBEDDING_CHANNELS = frozenset({"chunk", "fact", "entity"})


@dataclass(frozen=True)
class PreparedChunk:
    """Chunk content and vector data prepared for transactional persistence."""

    id: str
    message_index: int
    chunk_index: int
    role: str
    content: str
    source_timestamp: int | None
    created_at: str
    embedding: bytes
    embedding_dimension: int
    embedding_model: str
    start_char: int | None = None
    end_char: int | None = None


class ChunkCommit(int):
    """Chunk write count with a flag indicating whether new rows were inserted."""

    def __new__(cls, count: int, inserted: bool):
        value = super().__new__(cls, count)
        value.inserted = inserted
        return value


class SQLiteStore:
    def __init__(self, path: str | Path, embedding_identities: Mapping[str, str]):
        self.path = Path(path)
        if set(embedding_identities) != EMBEDDING_CHANNELS:
            raise ValueError(
                "Require chunk, fact and entity embedding identities"
            )
        if any(not isinstance(v, str) or not v for v in embedding_identities.values()):
            raise ValueError("Embedding identities must be non-empty strings")
        self.embedding_identities = dict(embedding_identities)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if existing_tables:
                if existing_tables != APPLICATION_TABLES:
                    raise ValueError(
                        "Unsupported database schema; no legacy migration is provided"
                    )
                stored_keys = {
                    row[0]
                    for row in connection.execute("SELECT key FROM store_metadata")
                }
                required_keys = {"schema_version"} | {
                    f"{channel}_embedding_identity" for channel in EMBEDDING_CHANNELS
                }
                if not required_keys <= stored_keys:
                    raise ValueError(
                        "Existing database has incomplete identity metadata"
                    )
            if existing_tables:
                version = connection.execute(
                    "SELECT value FROM store_metadata WHERE key='schema_version'"
                ).fetchone()
                if version is None or version[0] != SCHEMA_VERSION:
                    raise ValueError(
                        "Database schema version mismatch; use a new database (no migration)"
                    )
                for table, required in {
                    "raw_chunks": {
                        "speaker",
                        "source_timestamp",
                        "message_id",
                        "start_char",
                        "end_char",
                    },
                    "raw_messages": {"speaker", "source_timestamp", "fact_context_seq"},
                }.items():
                    columns = {
                        r[1] for r in connection.execute(f"PRAGMA table_info({table})")
                    }
                    if not required <= columns:
                        raise ValueError(
                            f"Database missing repair context columns: {table}"
                        )
            connection.execute("PRAGMA journal_mode = WAL")
            try:
                # All schema/metadata writes commit together, including first startup.
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + files("agentmem").joinpath("schema.sql").read_text("utf-8")
                )
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    **{
                        f"{k}_embedding_identity": v
                        for k, v in self.embedding_identities.items()
                    },
                }
                for key, value in metadata.items():
                    existing = connection.execute(
                        "SELECT value FROM store_metadata WHERE key = ?", (key,)
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
                            (key, value),
                        )
                    elif existing["value"] != value:
                        raise ValueError(f"Database metadata mismatch: {key}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _replay_count(
        connection: sqlite3.Connection,
        request_id: str,
        user_id: str,
        session_id: str,
        digest: str,
    ) -> int | None:
        existing = connection.execute(
            "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if existing is None:
            return None
        if (
            existing["user_id"] != user_id
            or existing["session_id"] != session_id
            or existing["payload_hash"] != digest
        ):
            raise RequestConflictError(
                "request_id was already used with a different payload"
            )
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM raw_chunks WHERE request_id = ?", (request_id,)
            ).fetchone()[0]
        )

    def replay_count(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[MemoryMessage],
    ) -> int | None:
        """None means new; an integer means exact replay (do not run derived channels)."""
        with self.connection() as connection:
            return self._replay_count(
                connection,
                request_id,
                user_id,
                session_id,
                payload_hash(user_id, session_id, messages),
            )

    def commit_chunks(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[MemoryMessage],
        chunks: Sequence[PreparedChunk],
    ) -> int:
        """Persist prepared chunks atomically and recheck request idempotency.

        The in-transaction check handles concurrent retries. Fact writes use
        their own transaction path.
        """
        digest = payload_hash(user_id, session_id, messages)
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                count = self._replay_count(
                    connection, request_id, user_id, session_id, digest
                )
                if count is not None:
                    connection.commit()
                    return ChunkCommit(count, False)
                if not chunks:
                    raise ValueError("messages produced no searchable content")
                now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                connection.execute(
                    "INSERT INTO add_requests VALUES (?, ?, ?, ?, ?)",
                    (request_id, user_id, session_id, digest, now),
                )
                for index, message in enumerate(messages):
                    connection.execute(
                        """INSERT INTO raw_messages(id,request_id,user_id,session_id,message_index,role,content,speaker,source_timestamp)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            message_id(request_id, index),
                            request_id,
                            user_id,
                            session_id,
                            index,
                            message.role,
                            message.content,
                            message.speaker,
                            message.timestamp,
                        ),
                    )
                for chunk in chunks:
                    original = messages[chunk.message_index].content
                    start = (
                        chunk.start_char
                        if chunk.start_char is not None
                        else original.find(chunk.content)
                    )
                    end = (
                        chunk.end_char
                        if chunk.end_char is not None
                        else start + len(chunk.content)
                    )
                    if (
                        start < 0
                        or end <= start
                        or original[start:end] != chunk.content
                    ):
                        raise ValueError("Chunk offsets do not match original message")
                    if chunk.embedding_model != self.embedding_identities["chunk"]:
                        raise ValueError(
                            "Chunk embedding identity does not match database"
                        )
                    connection.execute(
                        """INSERT INTO raw_chunks(
                            id, request_id, user_id, session_id, message_index, chunk_index,
                            role, content, source_timestamp, created_at,
                            embedding, embedding_dimension, embedding_model, speaker,
                            message_id,start_char,end_char
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.id,
                            request_id,
                            user_id,
                            session_id,
                            chunk.message_index,
                            chunk.chunk_index,
                            chunk.role,
                            chunk.content,
                            chunk.source_timestamp,
                            chunk.created_at,
                            chunk.embedding,
                            chunk.embedding_dimension,
                            chunk.embedding_model,
                            messages[chunk.message_index].speaker,
                            message_id(request_id, chunk.message_index),
                            start,
                            end,
                        ),
                    )
                connection.commit()
                return ChunkCommit(len(chunks), True)
            except Exception:
                connection.rollback()
                raise

    def chunks_for_user(self, user_id: str) -> list[sqlite3.Row]:
        # Restrict the candidate chunks to this user before retrieval ranking.
        with self.connection() as connection:
            return connection.execute(
                "SELECT * FROM raw_chunks WHERE user_id = ? ORDER BY rowid", (user_id,)
            ).fetchall()
