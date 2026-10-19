"""Rank-based result fusion with deterministic tie ordering across result types."""

from collections import defaultdict

TYPE_ORDER = {"chunk": 0, "fact": 1}


def rrf(ranked_lists, damping=60):
    # agent-memory-challenge src/index.py::rrf, adapted to hashable typed IDs.
    fused = defaultdict(float)
    for positions, weight in ranked_lists:
        if weight <= 0:
            continue
        for rank, position in enumerate(positions, start=1):
            fused[position] += weight / (damping + rank)
    return dict(fused)


def fuse(channels, top_k):
    records, ranks, lists = {}, {}, []
    for kind in TYPE_ORDER:
        hits = channels.get(kind, [])
        if not hits:
            continue
        keys = []
        for rank, hit in enumerate(hits, 1):
            key = (kind, str(hit.id))
            keys.append(key)
            records.setdefault(key, hit)
            ranks.setdefault(key, rank)
        lists.append((keys, 1.0))
    scores = rrf(lists)
    keys = sorted(
        scores, key=lambda key: (-scores[key], TYPE_ORDER[key[0]], ranks[key], key[1])
    )
    return [
        (kind, records[(kind, identifier)], scores[(kind, identifier)])
        for kind, identifier in keys[:top_k]
    ]
