"""Shared chunk candidate projection and weighted evidence RRF."""

from dataclasses import asdict

from .chunks import ChunkHit, timestamp
from .diagnostics import emit, error_fields, stage
from .provenance import source_chunks_for_user
from .schemas import SearchResponse


def search_evidence(service, payload):
    config, store = service.retrieval, service.chunk.store
    depth, memory_depth, source_depth = (
        max(50, payload.top_k),
        max(20, payload.top_k),
        max(100, payload.top_k),
    )
    enabled = {
        kind: bool(
            service.channels.get(kind, True)
            and getattr(config, kind + "_assist")
            and getattr(config, kind + "_weight") > 0
            and getattr(service, kind) is not None
        )
        for kind in ("fact",)
    }
    emit(
        "retrieval_configuration",
        **config.model_dump(),
        effective_assists=enabled,
        chunk_depth=depth,
        memory_depth=memory_depth,
        source_depth=source_depth,
    )
    with stage("channel_search", channel="chunk"):
        full = service.chunk.search(
            payload.user_id, payload.query, None, speaker_aware=True, bm25_fallback=True
        )
    full_rank = {hit.id: rank for rank, hit in enumerate(full, 1)}
    hits = {hit.id: hit for hit in full}
    rankings = {"chunk": {hit.id: rank for rank, hit in enumerate(full[:depth], 1)}}
    supports = {"fact": {}}
    for kind in ("fact",):
        rankings[kind] = {}
        if not enabled[kind]:
            continue
        try:
            with stage("channel_search", channel=kind):
                chunks_by_memory = source_chunks_for_user(
                    store, payload.user_id
                )
                eligible_ids = set(chunks_by_memory)
                memories = getattr(service, kind).search(
                    payload.user_id,
                    payload.query,
                    memory_depth,
                    eligible_ids=eligible_ids,
                )
                projected, support, raw_count, unsupported = {}, {}, 0, 0
                seen_chunks = set()
                for memory in memories:
                    chunks = chunks_by_memory.get(str(memory.id), [])
                    if not chunks:
                        unsupported += 1
                    for chunk_id in sorted(
                        chunks, key=lambda c: (full_rank.get(c, float("inf")), c)
                    ):
                        raw_count += 1
                        seen_chunks.add(chunk_id)
                        if chunk_id not in projected:
                            if len(projected) >= source_depth:
                                continue
                            projected[chunk_id] = len(projected) + 1
                        ids = support.setdefault(chunk_id, [])
                        typed_id = f"{kind}:{memory.id}"
                        if typed_id not in ids:
                            ids.append(typed_id)
                extra_hits = {}
                missing = set(projected) - hits.keys()
                if missing:
                    with store.connection() as conn:
                        placeholders = ",".join("?" for _ in missing)
                        rows = conn.execute(
                            f"SELECT * FROM raw_chunks WHERE user_id=? AND id IN ({placeholders})",
                            [payload.user_id, *sorted(missing)],
                        ).fetchall()
                        for row in rows:
                            extra_hits[row["id"]] = ChunkHit(
                                row["id"],
                                row["content"],
                                0.0,
                                timestamp(row["source_timestamp"])
                                if row["source_timestamp"] is not None
                                else row["created_at"],
                                row["role"],
                                row["session_id"],
                                row["source_timestamp"],
                                row["speaker"],
                            )
                hits.update(extra_hits)
                rankings[kind], supports[kind] = projected, support
                emit(
                    "source_projection",
                    channel=kind,
                    eligible_memory_count=len(eligible_ids),
                    memory_count=len(memories),
                    memory_ids=[str(m.id) for m in memories],
                    unsupported_count=unsupported,
                    raw_chunk_count=raw_count,
                    unique_chunk_count=len(projected),
                    duplicate_count=raw_count - len(seen_chunks),
                    truncated_unique_count=len(seen_chunks) - len(projected),
                    chunk_ids=list(projected),
                )
        except Exception as error:  # noqa: BLE001 -- each optional assist degrades independently
            rankings[kind], supports[kind] = {}, {}
            emit("assist_failed", channel=kind, **error_fields(error))
    candidates = set().union(*(set(ranks) for ranks in rankings.values()))
    contributions = {
        cid: {
            kind: getattr(config, kind + "_weight") / (60 + ranks[cid])
            if cid in ranks
            else 0.0
            for kind, ranks in rankings.items()
        }
        for cid in candidates
        if cid in hits
    }
    scores = {cid: sum(values.values()) for cid, values in contributions.items()}
    ordered = sorted(
        scores,
        key=lambda cid: (
            -scores[cid],
            full_rank.get(cid, float("inf")),
            rankings["fact"].get(cid, float("inf")),
            cid,
        ),
    )
    results, details = [], []
    for cid in ordered[: payload.top_k]:
        values = asdict(hits[cid])
        common = {key: values.pop(key) for key in ("content", "created_at")}
        hybrid_score = values.pop("score")
        values.pop("id")
        paths = [kind for kind in rankings if cid in rankings[kind]]
        support_ids = list(
            dict.fromkeys(
                mid
                for kind in ("fact",)
                for mid in supports[kind].get(cid, [])
            )
        )
        results.append(
            dict(
                id=f"chunk:{cid}",
                type="chunk",
                score=scores[cid],
                **common,
                chunk=values,
                retrieval_sources=paths,
                supporting_memory_ids=support_ids,
            )
        )
        details.append(
            {
                "chunk_id": cid,
                "ranks": {k: r.get(cid) for k, r in rankings.items()},
                "direct_full_rank": full_rank.get(cid),
                "contributions": contributions[cid],
                "score": scores[cid],
                "hybrid_score": hybrid_score,
                "supporting_memory_ids": support_ids,
            }
        )
    novel = candidates - rankings["chunk"].keys()
    emit(
        "evidence_candidates",
        candidate_ids=ordered,
        candidate_count=len(ordered),
        added_candidate_count=len(novel),
        added_selected_count=sum(c in novel for c in ordered[: payload.top_k]),
    )
    emit("evidence_fusion", results=details)
    emit(
        "fusion_result",
        result_count=len(results),
        typed_ids=[r["id"] for r in results],
        content=results,
    )
    return SearchResponse(data=results)
