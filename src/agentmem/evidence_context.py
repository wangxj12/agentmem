"""Expose the existing native answer excerpts through AML's content field."""

import json
from datetime import UTC, datetime

from .schemas import SearchResponse


SEARCH_CONTENT_FORMAT = "native-excerpt-v1"


def with_evidence_context(response: SearchResponse) -> SearchResponse:
    """Match legacy_answer_prompts.native_messages(..., 'agentmem', ...).

    Keep every selected excerpt, full text and native provenance. Internal
    identifiers and ranking diagnostics are not part of the answer context.
    Original result fields remain available outside the content string.
    """
    results, sessions = [], {}
    for rank, hit in enumerate(response.data, 1):
        excerpt = {
            "excerpt": rank,
            "text": hit.content,
            "created_at": hit.created_at,
            "type": hit.type,
        }
        chunk = getattr(hit, "chunk", None)
        if chunk is not None:
            for field in ("role", "speaker"):
                value = getattr(chunk, field)
                if value is not None:
                    excerpt[field] = value
            if chunk.source_timestamp is not None:
                excerpt["message_time"] = datetime.fromtimestamp(
                    chunk.source_timestamp / 1000, UTC
                ).isoformat()
            if chunk.session_id:
                excerpt["session"] = sessions.setdefault(
                    chunk.session_id, f"S{len(sessions) + 1}"
                )
        results.append(hit.model_copy(update={
            "content": json.dumps(excerpt, ensure_ascii=False, allow_nan=False)
        }))
    return SearchResponse(data=results)
