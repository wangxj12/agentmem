"""Fact extraction and retrieval orchestration with source-grounded persistence."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ._mem0.engine import FactEngine
from ._mem0.entity_extraction import extract_entities
from ._mem0.lemmatization import lemmatize_for_bm25
from ._mem0.scoring import score_and_rank
from .diagnostics import emit, error_fields
from .embeddings import Embedder, embedding_identity
from .fact_store import SQLiteFactMessages, SQLiteFactVectors
from .requests import MemoryMessage, message_id
from .store import SQLiteStore

logger = logging.getLogger(__name__)


class Mem0EmbeddingAdapter:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed(self, text, memory_action):
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action):
        if memory_action == "search":
            return self.embedder.encode_queries(texts)
        return self.embedder.encode_documents(texts)


@dataclass(frozen=True)
class FactHit:
    id: str
    content: str
    score: float
    created_at: str
    linked_memory_ids: list[str]
    attributed_to: str | None


class FactChannel:
    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        llm,
        *,
        message_context=True,
        grounded_extraction=True,
        extraction_config=None,
    ):
        self.extraction_config = extraction_config
        self.message_context = message_context
        self.grounded_extraction = grounded_extraction
        if any(
            embedding_identity(embedder) != store.embedding_identities[channel]
            for channel in ("fact", "entity")
        ):
            raise ValueError("Fact/entity embedding identity does not match database")
        self.store = store
        self.embedding_model = Mem0EmbeddingAdapter(embedder)
        self.llm = llm
        self.vector_store = SQLiteFactVectors(store)
        self.entity_store = SQLiteFactVectors(store, entities=True)
        self.messages = SQLiteFactMessages(store)

    def _engine(self, observation_date=None):
        return FactEngine(
            self.vector_store,
            self.entity_store,
            self.messages,
            self.embedding_model,
            self.llm,
            observation_date,
            message_context=self.message_context,
            grounded_extraction=self.grounded_extraction,
            provenance=True,
            extraction_config=self.extraction_config,
        )

    def add(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[MemoryMessage],
    ):
        # challenge store.add filters blank content, keeps role/content, takes min timestamp.
        stamps = [
            message.timestamp for message in messages if message.timestamp is not None
        ]
        observation_date = None
        if stamps:
            value = min(stamps)
            # challenge Mem0Backend.add: preserve seconds; convert millisecond epochs.
            seconds = int(value / 1000) if abs(value) > 1e11 else int(value)
            # Convert message timestamps to UTC dates for the prompt's observation date.
            observation_date = (
                datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()
            )
        filtered = [
            {
                "role": message.role,
                "content": message.content,
                "name": message.speaker,
                "source_timestamp": message.timestamp,
                "source_message_id": message_id(request_id, index),
            }
            for index, message in enumerate(messages)
            if message.content and message.content.strip()
        ]
        # Persist metadata even in legacy prompt mode; formatting is independently switchable.
        return self._engine(observation_date)._add_to_vector_store(
            filtered,
            {"request_id": request_id, "user_id": user_id, "session_id": session_id},
            {"user_id": user_id},
            True,
        )

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        *,
        eligible_ids: set[str] | None = None,
    ) -> list[FactHit]:
        # challenge Mem0Backend.search: a failing fact channel is an empty channel.
        try:
            return self._search(user_id, query, top_k, eligible_ids=eligible_ids)
        except Exception as error:
            emit("fact_search_failed", channel="fact", **error_fields(error))
            logger.exception(
                "Fact search failed; continuing with other channels",
                extra={"channel": "fact"},
            )
            return []

    def _search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        *,
        eligible_ids: set[str] | None = None,
    ) -> list[FactHit]:
        if eligible_ids is not None:
            eligible_ids = {str(value) for value in eligible_ids}
            if not eligible_ids:
                return []
        lemmatize_for_bm25(query)  # Mem0 preprocessing still runs for the FAISS path.
        entities = extract_entities(query)
        embedding = self.embedding_model.embed(query, "search")
        filters = {"user_id": user_id}
        semantic = self.vector_store.search(
            query=query,
            vectors=embedding,
            top_k=max(top_k * 4, 60),
            filters=filters,
            allowed_ids=eligible_ids,
        )
        boosts = (
            self._engine()._compute_entity_boosts(entities, filters) if entities else {}
        )
        scored = score_and_rank(
            [
                {"id": item.id, "score": item.score, "payload": item.payload}
                for item in semantic
            ],
            bm25_scores={},
            entity_boosts=boosts,
            threshold=0.0,
            top_k=top_k,
        )
        emit(
            "fact_ranking",
            channel="fact",
            candidate_count=len(semantic),
            content={
                "semantic": [{"id": x.id, "score": x.score} for x in semantic],
                "entities": entities,
                "boosts": boosts,
                "ranked": scored,
            },
        )
        results = []
        for item in scored:
            payload = item.get("payload") or {}
            text = payload.get("data") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            results.append(
                FactHit(
                    id=str(item["id"]),
                    content=text.strip(),
                    score=item["score"],
                    created_at=payload.get("created_at") or payload.get("updated_at"),
                    linked_memory_ids=payload.get("linked_memory_ids", []),
                    attributed_to=payload.get("attributed_to"),
                )
            )
        return results
