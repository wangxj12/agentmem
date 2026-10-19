"""Request identity and payload serialization helpers."""

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryMessage:
    role: str
    content: str
    timestamp: int | None = None
    speaker: str | None = None


class RequestConflictError(ValueError):
    pass


def payload_hash(user_id: str, session_id: str, messages: list[MemoryMessage]) -> str:
    """vanilla memory_store.SQLiteMemoryStore._payload_hash, verbatim serialization."""
    value = {
        "user_id": user_id,
        "session_id": session_id,
        "messages": [
            {
                k: v
                for k, v in message.__dict__.items()
                if k != "speaker" or v is not None
            }
            for message in messages
        ],
    }
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def message_id(request_id: str, index: int) -> str:
    value = json.dumps([request_id, index], ensure_ascii=False, separators=(",", ":"))
    return "msg_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
