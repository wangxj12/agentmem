"""SQLite adapters for fact, entity, and message persistence.

FAISS is an in-memory calculator built from scoped SQLite rows once per search
batch; no FAISS or Mem0 database or cross-request cache is created. Candidate
over-fetch/filter ordering matches FAISS 1.15.
"""

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import faiss
import numpy as np

from .diagnostics import stage
from .provenance import insert_sources
from .store import SQLiteStore


@dataclass
class VectorResult:
    id: str
    score: float | None
    payload: dict


class SQLiteFactVectors:
    def __init__(self, store: SQLiteStore, *, entities: bool = False):
        self.store = store
        self.entities = entities
        self.table = "fact_entities" if entities else "facts"
        self.identity = store.embedding_identities["entity" if entities else "fact"]

    def _rows(self, filters=None):
        # Push down only the indexed user scope. Keep payload filtering below
        # for all other fields and for the original Python equality semantics.
        sql = f"SELECT * FROM {self.table}"
        params = ()
        user_id = (filters or {}).get("user_id")
        if isinstance(user_id, str):
            sql += " WHERE user_id=?"
            params = (user_id,)
        sql += " ORDER BY rowid"
        with stage("fact_store_read", channel="fact", table=self.table) as metrics:
            with self.store.connection() as connection:
                rows = connection.execute(sql, params).fetchall()
            metrics["row_count"] = len(rows)
        return rows

    def _filtered_rows(self, filters, allowed_ids=None):
        rows = self._rows(filters)
        if not filters and allowed_ids is None:
            return rows
        selected = []
        for row in rows:
            if allowed_ids is not None and str(row["id"]) not in allowed_ids:
                continue
            if not filters:
                selected.append(row)
                continue
            payload = self._payload(row)
            if all(payload.get(k) == v for k, v in filters.items()):
                selected.append(row)
        return selected

    def _payload(self, row):
        if self.entities:
            return {
                "data": row["entity_text"],
                "entity_type": row["entity_type"],
                "linked_memory_ids": json.loads(row["linked_fact_ids"]),
                "user_id": row["user_id"],
            }
        result = {
            "data": row["content"],
            "hash": row["content_hash"],
            "text_lemmatized": row["text_lemmatized"],
            "linked_memory_ids": json.loads(row["linked_memory_ids"]),
            "user_id": row["user_id"],
            "request_id": row["request_id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if row["attributed_to"] is not None:
            result["attributed_to"] = row["attributed_to"]
        return result

    def insert(self, vectors, ids, payloads):
        # FAISS.insert copies float32 and normalizes before indexing cosine vectors.
        matrix = np.array(vectors, dtype=np.float32)
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)
        faiss.normalize_L2(matrix)
        with self.store.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                for vector, vector_id, payload in zip(matrix, ids, payloads):
                    now = datetime.now(UTC).isoformat()
                    data = payload["data"]
                    blob = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
                    if self.entities:
                        connection.execute(
                            """INSERT INTO fact_entities(
                                id,user_id,entity_text,normalized_text,entity_type,linked_fact_ids,
                                embedding,embedding_dimension,embedding_model,created_at,updated_at
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                vector_id,
                                payload["user_id"],
                                data,
                                " ".join(data.strip().lower().split()),
                                payload["entity_type"],
                                json.dumps(payload.get("linked_memory_ids", [])),
                                blob,
                                len(vector),
                                self.identity,
                                now,
                                now,
                            ),
                        )
                    else:
                        connection.execute(
                            """INSERT INTO facts(
                                id,request_id,user_id,session_id,content,content_hash,text_lemmatized,
                                linked_memory_ids,attributed_to,created_at,updated_at,
                                embedding,embedding_dimension,embedding_model
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                vector_id,
                                payload["request_id"],
                                payload["user_id"],
                                payload["session_id"],
                                data,
                                payload["hash"],
                                payload["text_lemmatized"],
                                json.dumps(payload.get("linked_memory_ids", [])),
                                payload.get("attributed_to"),
                                payload["created_at"],
                                payload["updated_at"],
                                blob,
                                len(vector),
                                self.identity,
                            ),
                        )
                    if not self.entities:
                        insert_sources(
                            connection,
                            payload["user_id"],
                            payload.get("_sources", []),
                            fact_id=vector_id,
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def merge_sources(self, memory_id, user_id, sources):
        with self.store.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                insert_sources(connection, user_id, sources, fact_id=memory_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _prepare_search(self, filters, allowed_ids=None):
        rows = self._filtered_rows(filters, allowed_ids)
        if not rows:
            return rows, None
        with stage(
            "fact_index_build", channel="fact", table=self.table, row_count=len(rows)
        ):
            dimension = rows[0]["embedding_dimension"]
            matrix = np.vstack(
                [
                    np.frombuffer(row["embedding"], dtype=np.float32, count=dimension)
                    for row in rows
                ]
            )
            index = faiss.IndexFlatIP(dimension)
            index.add(matrix)
        return rows, index

    def _search_prepared(self, rows, index, vectors, top_k, filters):
        if index is None:
            return []
        queries = np.array(vectors, dtype=np.float32)
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        faiss.normalize_L2(queries)
        fetch_k = top_k * 2 if filters else top_k
        scores, positions = index.search(queries, fetch_k)
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position == -1:
                continue
            row = rows[int(position)]
            payload = self._payload(row)
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            results.append(VectorResult(str(row["id"]), float(score), payload))
            if len(results) >= top_k:
                break
        return results

    def search(self, query, vectors, top_k=5, filters=None, allowed_ids=None):
        rows, index = self._prepare_search(filters, allowed_ids)
        with stage(
            "fact_vector_search", channel="fact", table=self.table, query_count=1
        ):
            return self._search_prepared(rows, index, vectors, top_k, filters)

    def search_batch(self, queries, vectors_list, top_k=5, filters=None, on_error=None):
        pairs = list(zip(queries, vectors_list))
        if not pairs:
            return []
        # One batch-local snapshot and index; later calls see newly committed rows.
        rows, index = self._prepare_search(filters)
        with stage(
            "fact_vector_search",
            channel="fact",
            table=self.table,
            query_count=len(pairs),
        ):
            # Keep per-query FAISS behavior while reusing the batch-local index.
            results = []
            for query, vector in pairs:
                try:
                    results.append(
                        self._search_prepared(rows, index, vector, top_k, filters)
                    )
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(query, error)
                    results.append([])
            return results

    def keyword_search(self, **kwargs):
        # Mem0 VectorStoreBase default, inherited by the FAISS backend.
        return None

    def list(self, filters=None, top_k=100):
        result = []
        for row in self._rows(filters):
            payload = self._payload(row)
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            result.append(VectorResult(str(row["id"]), None, payload))
            if len(result) >= top_k:
                break
        return [result]

    def update(self, vector_id, vector=None, payload=None):
        # Entity linking updates linked-fact metadata without changing its vector.
        if not self.entities or vector is not None:
            raise ValueError("Vector updates are unsupported; update entity metadata via payload")
        with self.store.connection() as connection:
            connection.execute(
                """UPDATE fact_entities SET linked_fact_ids=?,updated_at=?
                   WHERE id=? AND user_id=?""",
                (
                    json.dumps(payload.get("linked_memory_ids", [])),
                    datetime.now(UTC).isoformat(),
                    vector_id,
                    payload["user_id"],
                ),
            )
            connection.commit()


class SQLiteFactMessages:
    """Keep original messages; mark successful fact context in commit order."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._lock = threading.Lock()

    @staticmethod
    def _user(session_scope):
        if not session_scope.startswith("user_id="):
            raise ValueError("Fact context requires a user scope")
        return session_scope[len("user_id=") :]

    def save_messages(self, messages, session_scope):
        user_id = self._user(session_scope)
        with self._lock, self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = connection.execute(
                "SELECT COALESCE(MAX(fact_context_seq),0) FROM raw_messages"
            ).fetchone()[0]
            for message in messages:
                stable = message.get("source_message_id")
                if stable is None:
                    continue  # Reference-only callers have no registered provenance.
                row = connection.execute(
                    "SELECT content,fact_context_seq FROM raw_messages WHERE id=? AND user_id=?",
                    (stable, user_id),
                ).fetchone()
                if row is None or row["content"] != message["content"]:
                    raise ValueError(
                        "Fact context message is outside original user input"
                    )
                if row["fact_context_seq"] is None:
                    sequence += 1
                    connection.execute(
                        "UPDATE raw_messages SET fact_context_seq=? WHERE id=? AND user_id=?",
                        (sequence, stable, user_id),
                    )
            connection.commit()

    def get_last_messages(self, session_scope, limit=10):
        with self._lock, self.store.connection() as connection:
            rows = connection.execute(
                """SELECT role,content,speaker AS name,source_timestamp,
                          id AS source_message_id,fact_context_seq FROM raw_messages
                   WHERE user_id=? AND fact_context_seq IS NOT NULL
                   ORDER BY fact_context_seq DESC LIMIT ?""",
                (self._user(session_scope), limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]
