"""Validated message citations and a shared SQLite provenance relation."""

from .diagnostics import emit
from .requests import message_id

SOURCE_RULES = """
SOURCE CITATIONS:
Every extracted fact must include sources, an array of objects:
{"message_id":"m0","quote":"an exact verbatim excerpt","occurrence":1}.
Only use message_id values displayed on actual messages or visible history.
Existing memory IDs are NOT source message IDs. Quote the shortest sufficient
verbatim evidence, preserving spelling and punctuation. If an excerpt repeats
in that message, occurrence is its 1-based occurrence. Do not quote truncation
ellipsis or text outside the visible history. Never invent IDs or quotations.
No evidence means an empty source array, not a guessed citation.
"""


def insert_sources(conn, user_id, sources, *, fact_id):
    """Caller owns the same transaction as the derived record."""
    for source in sources:
        row = conn.execute(
            "SELECT start_char,end_char FROM raw_chunks WHERE id=? AND user_id=?",
            (source["chunk_id"], user_id),
        ).fetchone()
        if row is None or not (
            row["start_char"]
            <= source["source_start"]
            < source["source_end"]
            <= row["end_char"]
        ):
            raise ValueError("Source range is outside the user's chunk")
        conn.execute(
            """INSERT INTO memory_sources(user_id,fact_id,chunk_id,
                   source_role,source_start,source_end) VALUES(?,?,?,?,?,?)
               ON CONFLICT DO NOTHING""",
            (
                user_id,
                fact_id,
                source["chunk_id"],
                source["source_role"],
                source["source_start"],
                source["source_end"],
            ),
        )


class SourceContext:
    def __init__(self, store, user_id, request_id, messages, history=()):
        self.store, self.user_id = store, user_id
        self.allowed = {}
        self.messages = self._catalog(messages, request_id, "m", None)
        self.history = self._catalog(history, None, "h", 300)

    def _catalog(self, messages, request_id, prefix, limit):
        result = []
        with self.store.connection() as conn:
            for index, message in enumerate(messages):
                value = (
                    dict(message)
                    if isinstance(message, dict)
                    else dict(message.__dict__)
                )
                stable = value.get("source_message_id")
                if stable is None and request_id is not None:
                    stable = message_id(request_id, index)
                ref = f"{prefix}{index}"
                row = conn.execute(
                    "SELECT * FROM raw_messages WHERE id=? AND user_id=?",
                    (stable, self.user_id),
                ).fetchone()
                text = value.get("content", "")
                if row is not None and row["content"] == text:
                    self.allowed[ref] = (
                        dict(row),
                        text if limit is None else text[:limit],
                    )
                value["source_ref"] = ref
                value["source_message_id"] = stable
                result.append(value)
        return result

    def resolve(self, references, role, *, report=None):
        rejected = []
        sources = []
        valid_references = 0
        if not isinstance(references, list):
            rejected.append("missing_or_invalid_array")
            references = []
        with self.store.connection() as conn:
            for ref in references:
                if not isinstance(ref, dict) or not isinstance(
                    ref.get("message_id"), str
                ):
                    rejected.append("invalid_reference")
                    continue
                allowed = self.allowed.get(ref["message_id"])
                if allowed is None:
                    rejected.append("unknown_or_unavailable_message")
                    continue
                row, visible = allowed
                quote = ref.get("quote")
                if not isinstance(quote, str) or not quote.strip():
                    rejected.append("invalid_quote")
                    continue
                matches, at = [], visible.find(quote)
                while at >= 0:
                    matches.append(at)
                    at = visible.find(quote, at + 1)
                occurrence = ref.get("occurrence", 1 if len(matches) == 1 else None)
                if not matches:
                    rejected.append("quote_not_visible")
                    continue
                if type(occurrence) is not int or not 1 <= occurrence <= len(matches):
                    rejected.append("ambiguous_or_invalid_occurrence")
                    continue
                start, end = (
                    matches[occurrence - 1],
                    matches[occurrence - 1] + len(quote),
                )
                chunks = conn.execute(
                    """SELECT id,chunk_index,start_char,end_char FROM raw_chunks
                       WHERE message_id=? AND user_id=? ORDER BY chunk_index""",
                    (row["id"], self.user_id),
                ).fetchall()
                selected = [
                    c
                    for c in chunks
                    if c["start_char"] <= start and c["end_char"] >= end
                ][:1]
                if not selected:
                    cursor = start
                    while cursor < end:
                        covering = [
                            c
                            for c in chunks
                            if c["start_char"] <= cursor < c["end_char"]
                        ]
                        if not covering:
                            if visible[cursor].isspace():
                                cursor += 1
                                continue
                            selected = []
                            break
                        best = max(
                            covering, key=lambda c: (c["end_char"], -c["chunk_index"])
                        )
                        selected.append(best)
                        cursor = best["end_char"]
                if not selected:
                    rejected.append("quote_not_covered")
                    continue
                valid_references += 1
                for chunk in selected:
                    sources.append(
                        {
                            "chunk_id": chunk["id"],
                            "source_role": role,
                            "source_start": max(start, chunk["start_char"]),
                            "source_end": min(end, chunk["end_char"]),
                        }
                    )
        unique = {tuple(sorted(s.items())): s for s in sources}
        emit(
            "source_validation",
            source_role=role,
            reference_count=len(references),
            accepted_reference_count=valid_references,
            accepted_link_count=len(unique),
            rejected_count=len(rejected),
            rejected_reasons=rejected,
        )
        if report is not None:
            report.update(
                reference_count=len(references),
                accepted_reference_count=valid_references,
                rejected_count=len(rejected),
                rejected_reasons=rejected,
            )
        return list(unique.values())


def source_chunks_for_user(store, user_id):
    """Map source-qualified Fact IDs to their valid Base chunk IDs."""
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT s.fact_id AS memory_id,s.chunk_id
               FROM memory_sources s
               JOIN facts f ON f.id=s.fact_id AND f.user_id=s.user_id
               JOIN raw_chunks c ON c.id=s.chunk_id AND c.user_id=s.user_id
               WHERE s.user_id=? AND s.source_role='fact'
                 AND s.source_start >= c.start_char
                 AND s.source_end <= c.end_char
                 AND s.source_start < s.source_end
               ORDER BY s.id""",
            (user_id,),
        ).fetchall()
    result = {}
    for row in rows:
        memory_id, chunk_id = str(row["memory_id"]), str(row["chunk_id"])
        chunks = result.setdefault(memory_id, [])
        if chunk_id not in chunks:
            chunks.append(chunk_id)
    return result
