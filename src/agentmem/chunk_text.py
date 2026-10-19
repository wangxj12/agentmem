"""Exact vanilla token/chunk/BM25 functions, commit 31ab7bf9."""

import math
import re
from collections import Counter

import numpy as np

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if any(character.isalnum() or character == "_" for character in token)
    ]


def bm25_scores(
    documents: list[str], query: str, k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    tokenized = [lexical_tokens(document) for document in documents]
    query_terms = set(lexical_tokens(query))
    scores = np.zeros(len(documents), dtype=np.float32)
    if not documents or not query_terms:
        return scores

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    average_length = average_length or 1.0
    document_frequency = Counter(
        term for tokens in tokenized for term in query_terms.intersection(tokens)
    )
    document_count = len(documents)
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        length_normalization = 1.0 - b + b * len(tokens) / average_length
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + k1 * length_normalization)
            )
    return scores


def split_memory_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_tokens")
    matches = list(TOKEN_PATTERN.finditer(text))
    if not matches:
        return []
    chunks: list[str] = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(matches):
        end = min(start + chunk_tokens, len(matches))
        chunk = text[matches[start].start() : matches[end - 1].end()].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(matches):
            break
        start += step
    return chunks


def split_memory_spans(text: str, chunk_tokens: int, overlap_tokens: int):
    """Same token windows as split_memory_text, with original Unicode offsets."""
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("Invalid chunk parameters")
    matches = list(TOKEN_PATTERN.finditer(text))
    for start in range(0, len(matches), chunk_tokens - overlap_tokens):
        end = min(start + chunk_tokens, len(matches))
        lo, hi = matches[start].start(), matches[end - 1].end()
        yield lo, hi, text[lo:hi]
        if end == len(matches):
            break
