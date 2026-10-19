"""Vanilla-aligned chunk channel; source commit 31ab7bf9 (see README.md)."""

import hashlib
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from .chunk_text import bm25_scores, split_memory_spans
from .diagnostics import emit, error_fields
from .embeddings import Embedder, embedding_identity
from .requests import MemoryMessage
from .store import ChunkCommit, PreparedChunk, SQLiteStore


@dataclass(frozen=True)
class ChunkHit:
    id: str
    content: str
    score: float
    created_at: str
    role: str
    session_id: str
    source_timestamp: int | None
    speaker: str | None = None


def timestamp(value: int | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return (
        datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")
    )


class ChunkChannel:
    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        chunk_tokens: int = 320,
        overlap_tokens: int = 40,
    ):
        self.store = store
        self.embedder = embedder
        self.identity = embedding_identity(embedder)
        if self.identity != store.embedding_identities["chunk"]:
            raise ValueError(
                "Configured embedding model does not match the model used by this memory database"
            )
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._embedding_lock = threading.RLock()

    def add(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[MemoryMessage],
    ) -> int:
        count = self.store.replay_count(request_id, user_id, session_id, messages)
        if count is not None:
            return ChunkCommit(count, False)

        pending = []
        offsets = {}
        for message_index, message in enumerate(messages):
            for chunk_index, (lo, hi, content) in enumerate(
                split_memory_spans(
                    message.content, self.chunk_tokens, self.overlap_tokens
                )
            ):
                identity = f"{request_id}:{message_index}:{chunk_index}"
                memory_id = (
                    "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                )
                offsets[memory_id] = (lo, hi)
                pending.append(
                    (
                        memory_id,
                        message_index,
                        chunk_index,
                        message.role,
                        content,
                        message.timestamp,
                        timestamp(message.timestamp),
                    )
                )
        if not pending:
            raise ValueError("messages produced no searchable content")

        with self._embedding_lock:
            vectors = self.embedder.encode_documents([item[4] for item in pending])
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(pending):
            raise ValueError("Embedder returned an invalid document vector matrix")
        chunks = [
            PreparedChunk(
                *item,
                embedding=np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
                embedding_dimension=int(vector.shape[0]),
                embedding_model=self.identity,
                start_char=offsets[item[0]][0],
                end_char=offsets[item[0]][1],
            )
            for item, vector in zip(pending, vectors)
        ]
        return self.store.commit_chunks(
            request_id, user_id, session_id, messages, chunks
        )

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int | None,
        *,
        mode: str = "hybrid",
        speaker_aware: bool = False,
        bm25_fallback: bool = False,
    ) -> list[ChunkHit]:
        """Search stored chunks with dense or hybrid ranking.

        As in vanilla, the embedder supplies normalized vectors. Do not normalize
        here: vanilla's store uses float32 dot products without a second transform.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")
        rows = self.store.chunks_for_user(user_id)
        if not rows:
            return []
        documents = [
            f"{row['speaker']}: {row['content']}"
            if speaker_aware and row["speaker"]
            else str(row["content"])
            for row in rows
        ]
        lexical = bm25_scores(documents, query) if mode == "hybrid" else None
        try:
            with self._embedding_lock:
                query_vectors = np.asarray(
                    self.embedder.encode_queries([query]), dtype=np.float32
                )
            if query_vectors.ndim != 2 or query_vectors.shape[0] != 1:
                raise ValueError("Embedder returned an invalid query vector matrix")
            dimension = int(rows[0]["embedding_dimension"])
            if query_vectors.shape[1] != dimension:
                raise ValueError(
                    "Query embedding dimension does not match stored memories"
                )
            matrix = np.vstack(
                [
                    np.frombuffer(row["embedding"], dtype=np.float32, count=dimension)
                    for row in rows
                ]
            )
            dense_scores = (matrix @ query_vectors.T)[:, 0]
            if not np.isfinite(dense_scores).all():
                raise ValueError("Nonfinite dense score")
        except Exception as error:
            if not bm25_fallback or lexical is None:
                raise
            emit("chunk_bm25_fallback", channel="chunk", **error_fields(error))
            dense_scores = None
        if dense_scores is None:
            result_scores = lexical
            positions = np.flatnonzero(lexical > 0)
            positions = positions[np.argsort(-lexical[positions], kind="stable")]
        else:
            result_scores = dense_scores
            positions = np.argsort(-dense_scores, kind="stable")
        if mode == "hybrid" and dense_scores is not None:
            fused = np.zeros(len(rows), dtype=np.float32)
            for rank, position in enumerate(positions):
                fused[int(position)] += 1.0 / (60 + rank + 1)
            lexical_positions = np.flatnonzero(lexical > 0)
            lexical_positions = lexical_positions[
                np.argsort(-lexical[lexical_positions], kind="stable")
            ]
            for rank, position in enumerate(lexical_positions):
                fused[int(position)] += 1.0 / (60 + rank + 1)
            positions = np.lexsort((-dense_scores, -fused))
            result_scores = fused

        results = []
        for position in positions[:top_k]:
            row = rows[int(position)]
            created_at = (
                timestamp(row["source_timestamp"])
                if row["source_timestamp"] is not None
                else row["created_at"]
            )
            score = float(result_scores[int(position)])
            if not math.isfinite(score):
                score = 0.0
            results.append(
                ChunkHit(
                    id=str(row["id"]),
                    content=str(row["content"]),
                    score=score,
                    created_at=str(created_at),
                    role=str(row["role"]),
                    session_id=str(row["session_id"]),
                    source_timestamp=row["source_timestamp"],
                    speaker=row["speaker"],
                )
            )
        return results
