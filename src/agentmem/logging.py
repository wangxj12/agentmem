"""Structured channel logs for the CLI; no credentials or message bodies."""

import json
import logging
from datetime import UTC, datetime

from .diagnostics import scrub


class JsonFormatter(logging.Formatter):
    def format(self, record):
        value = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "diagnostic"):
            value.update(record.diagnostic)
        for key in ("event", "channel", "request_id"):
            if hasattr(record, key):
                value[key] = getattr(record, key)
        if record.exc_info:
            value["error_type"] = record.exc_info[0].__name__
        return scrub(json.dumps(value, ensure_ascii=False))


def configure_logging():
    logger = logging.getLogger("agentmem")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
