"""Format speaker and date context for extraction while preserving message text."""

import json
from datetime import UTC, datetime

CONTEXT_RULES = """
MESSAGE CONTEXT RULES:
- Each actual message and history message has its own source_date. Resolve
  yesterday/last week and other relative dates against THAT message's source_date,
  never the first date in a batch, the date of a later message, or today's date.
- A null source_date is unknown. Do not invent an absolute date for undated
  history or transfer another message's date to it.
- speaker identifies who is speaking; role is a transport role, not a person's
  name. Keep first-person facts attached to that speaker. Distinguish a quoted
  person from the speaker and from the person being addressed.
- History/existing memories provide context for new messages, not new events to
  extract again. Do not copy a historical relative date into a new date context.
"""


def message_record(message, *, seconds_compatible=False):
    value = message if isinstance(message, dict) else message.__dict__
    stamp = value.get("source_timestamp", value.get("timestamp"))
    source_date = None
    if stamp is not None:
        seconds = stamp if seconds_compatible and abs(stamp) <= 1e11 else stamp / 1000
        source_date = datetime.fromtimestamp(seconds, UTC).date().isoformat()
    return {
        **({"message_id": value["source_ref"]} if value.get("source_ref") else {}),
        "role": value.get("role"),
        "speaker": value.get("speaker") or value.get("name"),
        "source_timestamp": stamp,
        "source_date": source_date,
        "content": value.get("content"),
    }


def format_messages(messages, *, seconds_compatible=False, section="ACTUAL_MESSAGES"):
    records = [
        message_record(m, seconds_compatible=seconds_compatible) for m in messages
    ]
    return (
        f"BEGIN_{section}_JSON\n"
        + json.dumps(records, ensure_ascii=False)
        + f"\nEND_{section}_JSON"
    )


def format_history(messages):
    # Original Mem0 prompt takes dicts, formats and truncates content. Keep content
    # unchanged and put the small metadata header in the prompt-only role field.
    result = []
    for message in messages:
        record = message_record(message, seconds_compatible=True)
        header = {key: record[key] for key in ("role", "speaker", "source_date")}
        if "message_id" in record:
            header["message_id"] = record["message_id"]
        result.append(
            {
                "role": json.dumps(header, ensure_ascii=False),
                "content": record["content"],
            }
        )
    return result
