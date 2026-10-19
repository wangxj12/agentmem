"""Fact extraction pipeline with entity linking and search boosts."""

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from datetime import UTC, datetime

from ..config import FactExtractionConfig
from ..diagnostics import emit, observed, stage
from ..fact_protocol import repair_sources, response_format, validate_items
from ..grounding import GROUNDING_RULES
from ..message_context import CONTEXT_RULES, format_history, format_messages
from ..provenance import SOURCE_RULES, SourceContext
from .entity_extraction import extract_entities_batch
from .lemmatization import lemmatize_for_bm25
from .parsing import extract_json, parse_messages, remove_code_blocks
from .prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from .scoring import ENTITY_BOOST_WEIGHT

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _vector_store_list_rows(listed):
    if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list):
        return listed[0]
    if isinstance(listed, (list, tuple)):
        return listed
    return []


def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


class FactEngine:
    def __init__(
        self,
        vector_store,
        entity_store,
        db,
        embedding_model,
        llm,
        observation_date=None,
        *,
        message_context=False,
        grounded_extraction=False,
        provenance=False,
        extraction_config=None,
    ):
        self.extraction_config = extraction_config or FactExtractionConfig()
        self.provenance = provenance
        self.message_context = message_context
        self.grounded_extraction = grounded_extraction
        self.vector_store = vector_store
        self.entity_store = entity_store
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        self.observation_date = observation_date
        self.custom_instructions = None

    @staticmethod
    def _normalize_entity_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _existing_entities_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.entity_store.list(filters=filters, top_k=10000)
        except Exception as e:
            logger.debug(
                f"Exact entity lookup failed, falling back to semantic dedup: {e}"
            )
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self._normalize_entity_text(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
        return rows_by_text

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self._normalize_entity_text(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {
            k: v
            for k, v in filters.items()
            if k in ("user_id", "agent_id", "run_id") and v
        }
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = self.embedding_model.embed_batch(entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            def report_search_error(_query, error):
                logger.warning("Entity boost search failed for one entity: %s", error)

            matches_by_entity = self.entity_store.search_batch(
                entity_texts,
                embeddings,
                top_k=500,
                filters=search_filters,
                on_error=report_search_error,
            )
            for matches in matches_by_entity:
                for match in matches:
                    similarity = match.score if hasattr(match, "score") else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, "payload") else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (
                        1.0 + 0.001 * ((num_linked - 1) ** 2)
                    )
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(
                                memory_boosts.get(memory_key, 0.0), boost
                            )

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    @observed("fact_extract", channel="fact")
    def _add_to_vector_store(
        self, messages, metadata, filters, infer=True, prompt=None
    ):

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)
        source_context = None
        if self.provenance:
            source_context = SourceContext(
                self.vector_store.store,
                metadata["user_id"],
                metadata["request_id"],
                messages,
                last_messages,
            )
            messages, last_messages = source_context.messages, source_context.history

        # Phase 1: Existing memory retrieval
        search_filters = {
            k: v
            for k, v in filters.items()
            if k in ("user_id", "agent_id", "run_id") and v
        }
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append(
                {"id": str(idx), "text": mem.payload.get("data", "")}
            )

        # Phase 2: LLM extraction (single call)
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if self.grounded_extraction:
            system_prompt = (
                GROUNDING_RULES + "\n" + system_prompt + "\n" + GROUNDING_RULES
            )
        if self.message_context:
            system_prompt += "\n" + CONTEXT_RULES

        if self.provenance:
            system_prompt += "\n" + SOURCE_RULES
        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=(
                format_messages(messages, seconds_compatible=True)
                if self.message_context or self.provenance
                else parsed_messages
            ),
            last_k_messages=(
                format_history(last_messages)
                if self.message_context or self.provenance
                else last_messages
            ),
            custom_instructions=custom_instr,
            timestamp=self.observation_date,  # Approved date adaptation.
        )

        emit(
            "fact_prompt",
            channel="fact",
            history_count=len(last_messages),
            message_count=len(messages),
            content={"system": system_prompt, "user": user_prompt},
        )
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format(self.extraction_config.response_format),
            )
        except Exception as e:
            # Re-raise so callers can implement provider fallback / retry.
            # The original silent ``return []`` made upstream callers unable to
            # distinguish "LLM unavailable" (429/5xx/timeout) from "LLM
            # extracted no facts" -- both surfaced as an empty list.
            logger.error(
                "LLM extraction failed",
                extra={"event": "fact_llm_failed", "channel": "fact"},
            )
            raise LLMError("LLM extraction failed") from e

        emit("fact_llm_output", channel="fact", content={"response": response})
        # Parse response; diagnostics preserve the original tolerant behavior.
        parse_status = "parsed"
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                parse_status = "empty_response"
                extracted_memories = []
            else:
                try:
                    envelope = json.loads(response, strict=False)
                except json.JSONDecodeError:
                    envelope = json.loads(extract_json(response), strict=False)
                if not isinstance(envelope, dict):
                    raise TypeError("Expected a fact response object")
                if "memory" not in envelope:
                    emit(
                        "fact_protocol_error",
                        channel="fact",
                        reason="missing_memory_field",
                    )
                if set(envelope) - {"memory"}:
                    emit(
                        "fact_protocol_error",
                        channel="fact",
                        reason="unexpected_envelope_fields",
                    )
                extracted_memories = envelope.get("memory", [])
        except Exception as e:
            logger.error("Error parsing extraction response: %s", type(e).__name__)
            parse_status = "parse_error"
            extracted_memories = []

        emit(
            "fact_parse",
            channel="fact",
            parse_status=parse_status,
            raw_count=len(extracted_memories)
            if isinstance(extracted_memories, list)
            else None,
            content={"items": extracted_memories},
        )
        if source_context:
            extracted_memories = validate_items(extracted_memories)
            repair_sources(
                extracted_memories, source_context, self.llm, self.extraction_config
            )
        if not extracted_memories:
            emit("fact_accepted", channel="fact", accepted_count=0, discarded_count=0)
            # Save messages even if nothing extracted
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # Fallback: embed individually
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    emit(
                        "fact_embedding_failed",
                        channel="fact",
                        error_type=type(e).__name__,
                    )
                    logger.warning("Failed to embed memory text: %s", type(e).__name__)

        with stage("fact_records_prepare", channel="fact"):
            # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
            # Build set of existing hashes for dedup
            existing_hashes = {}
            for mem in existing_results:
                h = (
                    mem.payload.get("hash")
                    if hasattr(mem, "payload") and mem.payload
                    else None
                )
                if h:
                    existing_hashes[h] = mem.id

            records = []  # (memory_id, text, embedding, payload)
            seen_hashes = {}  # hash -> payload, for same-batch source union
            for mem in extracted_memories:
                text = mem.get("text")
                if not text:
                    continue

                mem_hash = hashlib.md5(text.encode()).hexdigest()
                sources = mem["_resolved_sources"] if source_context else []
                if self.provenance and not sources:
                    emit(
                        "memory_without_sources", channel="fact", content_hash=mem_hash
                    )
                if mem_hash in existing_hashes:
                    if self.provenance:
                        self.vector_store.merge_sources(
                            existing_hashes[mem_hash], metadata["user_id"], sources
                        )
                    continue
                if mem_hash in seen_hashes:
                    if self.provenance:
                        seen_hashes[mem_hash]["_sources"].extend(sources)
                    continue

                if text not in embed_map:
                    continue
                text_lemmatized = lemmatize_for_bm25(text)

                memory_id = str(uuid.uuid4())
                mem_metadata = deepcopy(metadata)
                mem_metadata["data"] = text
                seen_hashes[mem_hash] = mem_metadata
                if self.provenance:
                    mem_metadata["_sources"] = sources
                mem_metadata["text_lemmatized"] = text_lemmatized
                mem_metadata["hash"] = mem_hash
                raw_links = mem.get("linked_memory_ids", [])
                mem_metadata["linked_memory_ids"] = (
                    [
                        uuid_mapping[str(link)]
                        for link in raw_links
                        if isinstance(link, (str, int)) and str(link) in uuid_mapping
                    ]
                    if isinstance(raw_links, list)
                    else []
                )
                if "created_at" not in mem_metadata:
                    mem_metadata["created_at"] = datetime.now(UTC).isoformat()
                mem_metadata["updated_at"] = mem_metadata["created_at"]
                if mem.get("attributed_to"):
                    mem_metadata["attributed_to"] = mem["attributed_to"]

                records.append((memory_id, text, embed_map[text], mem_metadata))

            if not records:
                emit(
                    "fact_accepted",
                    channel="fact",
                    accepted_count=0,
                    discarded_count=len(extracted_memories),
                )
                self.db.save_messages(messages, session_scope)
                return []

            emit(
                "fact_accepted",
                channel="fact",
                accepted_count=len(records),
                discarded_count=len(extracted_memories) - len(records),
            )
        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        with stage("fact_records_write", channel="fact"):
            try:
                self.vector_store.insert(
                    vectors=all_vectors,
                    ids=all_ids,
                    payloads=all_payloads,
                )
            except Exception:
                if self.provenance:
                    raise  # Atomic memory/source persistence failure is observable.
                # Fallback: insert one by one
                for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                    try:
                        self.vector_store.insert(
                            vectors=[vec], ids=[mid], payloads=[pay]
                        )
                    except Exception as e:
                        emit(
                            "fact_persist_failed",
                            channel="fact",
                            memory_id=mid,
                            error_type=type(e).__name__,
                        )
                        logger.error(
                            "Failed to insert memory %s: %s", mid, type(e).__name__
                        )

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            with stage(
                "fact_entity_extract", channel="fact", text_count=len(all_texts)
            ):
                all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = self._normalize_entity_text(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(
                        entity_texts, "add"
                    )
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(
                                self.embedding_model.embed(t, "add")
                            )
                        except Exception:
                            entity_embeddings.append(None)

                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (
                        len(ordered_keys) - len(entity_embeddings)
                    )

                # Filter out entities with failed embeddings
                valid = [
                    (i, k)
                    for i, k in enumerate(ordered_keys)
                    if entity_embeddings[i] is not None
                ]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    with stage("fact_entity_exact_lookup", channel="fact"):
                        exact_matches = self._existing_entities_by_text(search_filters)

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    with stage("fact_entity_write", channel="fact"):
                        # 7d: Separate into inserts vs updates
                        to_insert_vectors, to_insert_ids, to_insert_payloads = (
                            [],
                            [],
                            [],
                        )
                        for j, key in enumerate(valid_keys):
                            entity_type, entity_text, memory_ids = global_entities[key]
                            matches = (
                                existing_matches[j] if j < len(existing_matches) else []
                            )
                            exact_match = exact_matches.get(key)

                            semantic_match = (
                                matches[0]
                                if matches and matches[0].score >= 0.95
                                else None
                            )
                            match = exact_match or semantic_match
                            if match:
                                # Update existing entity
                                payload = match.payload or {}
                                linked = set(payload.get("linked_memory_ids", []))
                                linked |= memory_ids
                                payload["linked_memory_ids"] = sorted(linked)
                                try:
                                    self.entity_store.update(
                                        vector_id=match.id,
                                        vector=None,
                                        payload=payload,
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Entity update failed for '{entity_text}': {e}"
                                    )
                            else:
                                # New entity — collect for batch insert
                                to_insert_vectors.append(valid_vectors[j])
                                to_insert_ids.append(str(uuid.uuid4()))
                                to_insert_payloads.append(
                                    {
                                        "data": entity_text,
                                        "entity_type": entity_type,
                                        "linked_memory_ids": sorted(memory_ids),
                                        **search_filters,
                                    }
                                )

                        # 7e: Single batch insert for all new entities
                        if to_insert_vectors:
                            try:
                                self.entity_store.insert(
                                    vectors=to_insert_vectors,
                                    ids=to_insert_ids,
                                    payloads=to_insert_payloads,
                                )
                            except Exception as e:
                                logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        with stage("fact_context_write", channel="fact"):
            self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"} for r in records
        ]

        return returned_memories
