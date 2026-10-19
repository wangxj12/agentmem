"""Independent tolerant channels with request-correlated, opt-in diagnostics."""

import hashlib
import logging
import time
from dataclasses import asdict

from .config import ChannelsConfig, FactRetryConfig, RetrievalConfig
from .diagnostics import Diagnostics, emit, observe_events, stage
from .evidence import search_evidence
from .evidence_context import SEARCH_CONTENT_FORMAT
from .fusion import fuse
from .requests import MemoryMessage
from .request_states import FactAttempt, RequestStates
from .schemas import AddResponse, SearchResponse

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        chunk,
        fact=None,
        channels=None,
        diagnostics=None,
        retrieval=None,
        fact_retry=None,
        configuration_id=None,
    ):
        self.chunk, self.fact = chunk, fact
        self.channels = ChannelsConfig.model_validate(channels or {}).model_dump()
        self.diagnostics = diagnostics or Diagnostics()
        self.retrieval = retrieval or RetrievalConfig()
        self.fact_retry = fact_retry or FactRetryConfig()
        self.states = RequestStates(chunk.store)
        self.configuration_id = configuration_id
        if self.retrieval.mode == "evidence" and not self.channels.get("chunk", True):
            raise ValueError("evidence retrieval requires channels.chunk=true")

    def _counts(self, request_id):
        # Count actual persisted rows rather than treating returned extraction
        # candidates as successful writes (Mem0 tolerates per-row failures).
        if self.diagnostics.path is None:
            return
        try:
            with self.chunk.store.connection() as conn:
                counts = {
                    name: conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE request_id=?",
                        (request_id,),
                    ).fetchone()[0]
                    for name, table in (
                        ("chunk", "raw_chunks"),
                        ("fact", "facts"),
                    )
                }
                emit("persisted_counts", counts=counts)
        except Exception:  # noqa: BLE001 -- diagnostic read failure is nonfatal
            emit("persisted_counts_unavailable")

    def add(self, payload):
        with self.diagnostics.scope(
            "add",
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        ):
            with self.states.locked(payload.request_id):
                return self._add(payload)

    def info(self):
        return {'database_id': self.states.database_id, 'configuration_id': self.configuration_id,
                'aml_search_content_format': SEARCH_CONTENT_FORMAT,
                'channels': self.channels, 'retrieval': self.retrieval.model_dump(),
                'fact_retry': self.fact_retry.model_dump(), 'request_status_version': 1}

    def _fact_add(self, payload, messages):
        old = self.states.get(payload.request_id)['channels']['fact']
        if old['status'] == 'succeeded':
            return
        # Replaying the unmodified engine cannot guarantee rebuilding entity links
        # after partial persistence. Keep this condition visible rather than claim
        # success just because dedup returns an empty list on a later attempt.
        sticky_partial = old.get('error_code') == 'partial_fact_write'
        engine_logger = logging.getLogger('agentmem._mem0.engine')
        all_warnings = list(old.get('warnings', []))
        expected = self.states.expected_hashes(payload.request_id)
        for attempt_number in range(self.fact_retry.max_retries + 1):
            self.states.set(payload.request_id, 'fact', 'running')
            observation = FactAttempt(lambda code, hashes=(): self.states.observe_failure(payload.request_id, code, hashes))
            engine_logger.addHandler(observation)
            try:
                with observe_events(observation.event), stage('channel_add', channel='fact'):
                    self.fact.add(payload.request_id, payload.user_id, payload.session_id, messages)
            except Exception as error:
                observation.exception(error)
                logger.exception('Derived write failed', extra={'event': 'add_channel_failed', 'channel': 'fact', 'request_id': payload.request_id})
            finally:
                engine_logger.removeHandler(observation)
            all_warnings.extend(observation.warnings)
            # Track candidates from an observed embedding failure only. A later
            # empty/dedup response must not hide facts dropped by an earlier call.
            if observation.error == 'fact_embedding_failed':
                expected |= observation.expected_hashes
            missing = self.states.missing_hashes(payload.user_id, expected)
            sticky_partial = sticky_partial or observation.partial
            code = 'partial_fact_write' if sticky_partial else observation.error
            if code is None and missing:
                code = 'fact_embeddings_incomplete'
            if code is None:
                self.states.set(payload.request_id, 'fact', 'succeeded', warnings=sorted(set(all_warnings)), expected_hashes=expected)
                return
            self.states.set(payload.request_id, 'fact', 'failed', error=code,
                            retryable=observation.retryable and not sticky_partial, warnings=sorted(set(all_warnings)), expected_hashes=expected)
            if not observation.retryable:
                break
            if attempt_number < self.fact_retry.max_retries:
                delay = self.fact_retry.backoff_seconds * (2 ** attempt_number)
                emit('fact_retry_wait', channel='fact', attempt=attempt_number + 1, delay_seconds=delay)
                time.sleep(delay)

    def _add(self, payload):
        messages = [MemoryMessage(**item.model_dump()) for item in payload.messages]
        enabled = self.fact is not None and self.channels.get('fact', True)
        self.states.register(payload, messages, enabled)
        try:
            self.states.set(payload.request_id, 'chunk', 'running')
            try:
                with stage("channel_add", channel="chunk") as metrics:
                    result = self.chunk.add(
                        payload.request_id, payload.user_id, payload.session_id, messages
                    )
                    metrics.update(inserted=result.inserted, count=int(result))
            except Exception as error:
                self.states.set(payload.request_id, 'chunk', 'failed', error=type(error).__name__)
                raise
            self.states.set(payload.request_id, 'chunk', 'succeeded')
            if enabled:
                self._fact_add(payload, messages)
            else:
                self.states.set(payload.request_id, 'fact', 'disabled')
        finally:
            self.states.finish(payload.request_id)
            self._counts(payload.request_id)
        return AddResponse(
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
            complete=self.states.get(payload.request_id)['complete'],
            channels=self.states.get(payload.request_id)['channels'],
        )

    def search(self, payload):
        with self.diagnostics.scope("search", user_id=payload.user_id):
            emit(
                "query",
                top_k=payload.top_k,
                query_sha256=hashlib.sha256(payload.query.encode()).hexdigest(),
                content={"query": payload.query},
            )
            return self._search(payload)

    def _search(self, payload):
        if self.retrieval.mode == "evidence":
            return search_evidence(self, payload)
        channels = {}
        for name, channel in [
            ("chunk", self.chunk),
            ("fact", self.fact),
        ]:
            if channel is None or not self.channels.get(name, True):
                continue
            try:
                with stage("channel_search", channel=name) as metrics:
                    channels[name] = channel.search(
                        payload.user_id, payload.query, payload.top_k
                    )
                    metrics["result_count"] = len(channels[name])
                    emit(
                        "channel_candidates",
                        channel=name,
                        candidate_ids=[str(hit.id) for hit in channels[name]],
                        content=[asdict(hit) for hit in channels[name]],
                    )
            except Exception:
                logger.exception(
                    "Search channel failed",
                    extra={"event": "search_channel_failed", "channel": name},
                )
        results = []
        for kind, hit, score in fuse(channels, payload.top_k):
            values = asdict(hit)
            common = {key: values.pop(key) for key in ("id", "content", "created_at")}
            values.pop("score")
            results.append(
                {
                    **common,
                    "id": f"{kind}:{hit.id}",
                    "type": kind,
                    "score": score,
                    kind: values,
                }
            )
        emit(
            "fusion_result",
            result_count=len(results),
            typed_ids=[hit["id"] for hit in results],
            content=results,
        )
        return SearchResponse(data=results)
