"""Resolve the inbound memory-system key without exposing it in errors."""

import os

from .config import APIConfig


def resolve_api_key(config: APIConfig) -> str | None:
    if not config.require_token:
        return None
    key = (config.api_key.get_secret_value() if config.api_key is not None
           else os.environ.get("AGENTMEM_API_KEY"))
    if not key or not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        raise ValueError(
            "Token authentication requires a nonempty ASCII key without whitespace; "
            "set AGENTMEM_API_KEY or api.api_key"
        )
    return key
