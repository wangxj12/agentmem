"""Batch status and retry observation; no changes to FactEngine algorithms."""

import json
import hashlib
import logging
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import uuid4

from .diagnostics import error_fields
from .requests import RequestConflictError, payload_hash


class RequestStates:
    def __init__(self, store):
        self.store = store
        self._guard = threading.Lock()
        self._locks = {}
        # The standard runner uses one service process. A previous process has
        # stopped before this runtime starts; interrupted work is safe to replay.
        with store.connection() as conn:
            conn.execute("INSERT OR IGNORE INTO store_metadata VALUES ('database_id', ?)", (uuid4().hex,))
            self.database_id = conn.execute("SELECT value FROM store_metadata WHERE key='database_id'").fetchone()[0]
            for channel in ('chunk', 'fact'):
                conn.execute(
                    f"UPDATE request_states SET {channel}_status='failed', {channel}_error=COALESCE({channel}_error,'interrupted'), updated_at=? WHERE {channel}_status='running'",
                    (datetime.now(UTC).isoformat(),),
                )
            conn.execute('UPDATE request_states SET processing=0 WHERE processing=1')
            conn.commit()

    @contextmanager
    def locked(self, request_id):
        with self._guard:
            entry = self._locks.setdefault(request_id, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if not entry[1]:
                    del self._locks[request_id]

    def register(self, payload, messages, fact_enabled):
        digest = payload_hash(payload.user_id, payload.session_id, messages)
        with self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                """INSERT OR IGNORE INTO request_states
                   (request_id,user_id,session_id,payload_hash,fact_status,updated_at)
                   VALUES(?,?,?,?,?,?)""",
                (payload.request_id, payload.user_id, payload.session_id, digest,
                 'pending' if fact_enabled else 'disabled', datetime.now(UTC).isoformat()),
            )
            row = conn.execute('SELECT * FROM request_states WHERE request_id=?', (payload.request_id,)).fetchone()
            if (row['user_id'], row['session_id'], row['payload_hash']) != (payload.user_id, payload.session_id, digest):
                conn.rollback()
                raise RequestConflictError('request_id was already used with a different payload')
            conn.execute('UPDATE request_states SET processing=1 WHERE request_id=?', (payload.request_id,))
            conn.commit()

    def finish(self, request_id):
        with self.store.connection() as conn:
            conn.execute('UPDATE request_states SET processing=0,updated_at=? WHERE request_id=?',
                         (datetime.now(UTC).isoformat(), request_id))
            conn.commit()

    def set(self, request_id, channel, status, *, error=None, retryable=True, warnings=(), expected_hashes=None):
        if channel not in ('chunk', 'fact'):
            raise ValueError('Unsupported status channel')
        columns = [f'{channel}_status=?', 'updated_at=?']
        values = [status, datetime.now(UTC).isoformat()]
        # Keep known partial failures during a retry, including a process crash.
        if status != 'running':
            columns.append(f'{channel}_error=?')
            values.append(error)
        if status == 'running':
            columns.append(f'{channel}_attempts={channel}_attempts+1')
        if channel == 'fact' and status != 'running':
            columns.extend(['fact_retryable=?', 'fact_warnings=?'])
            values.extend([int(retryable), json.dumps(list(warnings))])
            if expected_hashes is not None:
                columns.append('fact_expected_hashes=?')
                values.append(json.dumps(sorted(expected_hashes)))
        with self.store.connection() as conn:
            conn.execute(f"UPDATE request_states SET {','.join(columns)} WHERE request_id=?", (*values, request_id))
            conn.commit()

    def observe_failure(self, request_id, error, expected_hashes=()):
        """Checkpoint an already observed failure before later Fact writes."""
        with self.store.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT fact_expected_hashes FROM request_states WHERE request_id=?', (request_id,)).fetchone()
            expected = set(json.loads(row[0])) | set(expected_hashes)
            conn.execute("""UPDATE request_states SET
                         fact_error=CASE WHEN fact_error='partial_fact_write' THEN fact_error ELSE ? END,
                         fact_retryable=CASE WHEN ?='partial_fact_write' THEN 0 ELSE fact_retryable END,
                         fact_expected_hashes=?,updated_at=? WHERE request_id=?""",
                         (error, error, json.dumps(sorted(expected)), datetime.now(UTC).isoformat(), request_id))
            conn.commit()

    def expected_hashes(self, request_id):
        with self.store.connection() as conn:
            row = conn.execute('SELECT fact_expected_hashes FROM request_states WHERE request_id=?', (request_id,)).fetchone()
        return set(json.loads(row[0])) if row else set()

    def missing_hashes(self, user_id, expected):
        if not expected:
            return set()
        with self.store.connection() as conn:
            existing = {row[0] for row in conn.execute('SELECT content_hash FROM facts WHERE user_id=?', (user_id,))}
        return expected - existing

    def get(self, request_id):
        with self.store.connection() as conn:
            row = conn.execute('SELECT * FROM request_states WHERE request_id=?', (request_id,)).fetchone()
        if row is None:
            return None
        channels = {}
        for kind in ('chunk', 'fact'):
            channels[kind] = {'status': row[f'{kind}_status'], 'attempts': row[f'{kind}_attempts'],
                              'error_code': row[f'{kind}_error']}
        channels['fact'].update(retryable=bool(row['fact_retryable']), warnings=json.loads(row['fact_warnings']))
        return {'request_id': row['request_id'], 'user_id': row['user_id'], 'session_id': row['session_id'],
                'processing': bool(row['processing']),
                'complete': channels['chunk']['status'] == 'succeeded' and channels['fact']['status'] in ('succeeded', 'disabled'),
                'channels': channels}


class FactAttempt(logging.Handler):
    """Observe tolerated failures through existing events and engine logs."""
    def __init__(self, checkpoint=None):
        super().__init__(level=logging.DEBUG)
        self.thread_id = threading.get_ident()
        self.error = None
        self.retryable = True
        self.warnings = []
        self.partial = False
        self.expected_hashes = set()
        self.checkpoint = checkpoint

    def event(self, event, fields):
        if fields.get('channel') != 'fact':
            return
        if event == 'fact_parse':
            items = (fields.get('_content') or {}).get('items', [])
            if isinstance(items, list):
                self.expected_hashes = {
                    hashlib.md5(item['text'].encode()).hexdigest()
                    for item in items if isinstance(item, dict) and isinstance(item.get('text'), str) and item['text'].strip()
                }
        if event == 'fact_parse' and fields.get('parse_status') in ('parse_error', 'empty_response'):
            self.error = fields['parse_status']
        elif event == 'fact_protocol_error' and fields.get('reason') in ('missing_memory_field', 'memory_wrong_type', 'invalid_fact'):
            self.error = 'invalid_fact_response'
        elif event == 'fact_embedding_failed':
            self.error = 'fact_embedding_failed'
            if self.checkpoint:
                self.checkpoint(self.error, self.expected_hashes)
        elif event == 'fact_source_summary' and fields.get('final_with_sources', 0) < fields.get('fact_count', 0):
            # Preserve citation repair / source eligibility behavior, report quality.
            self.warnings.append('facts_without_valid_sources')
        elif event == 'fact_llm_metadata' and fields.get('finish_reason') == 'length':
            if fields.get('phase') == 'source_repair':
                self.warnings.append('source_repair_truncated')
            else:
                self.error = 'extraction_truncated'

    def emit(self, record):
        if record.thread != self.thread_id:
            return
        text = record.getMessage()
        if any(marker in text for marker in ('Batch entity linking failed', 'Batch entity insert failed', 'Entity update failed')):
            self.error = 'entity_linking_failed'
            self.partial = True
            if self.checkpoint:
                self.checkpoint('partial_fact_write')

    def exception(self, error):
        fields = error_fields(error)
        status = fields.get('http_status')
        self.error = f"http_{status}" if status else fields['error_type']
        self.retryable = status is None or status in (408, 429) or status >= 500
