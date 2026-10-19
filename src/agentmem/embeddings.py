"""Embedding interface used by the vanilla chunk channel."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def embedding_identity(embedder: Embedder) -> str:
    # vanilla SQLiteMemoryStore.__init__, context_messages=0.
    return str(
        getattr(
            embedder,
            "identity",
            getattr(
                embedder,
                "model_name",
                f"{embedder.__class__.__module__}.{embedder.__class__.__qualname__}",
            ),
        )
    )
