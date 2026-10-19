"""Smoke-test a running AgentMemory service using only the standard library.

Uses unique test users and writes a few memories to the configured database.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from uuid import uuid4


def request_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, object]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
        error.close()
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"{method} {url}: {error}") from error
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        preview = raw[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {status}, invalid JSON: {preview}") from error
    check(isinstance(body, dict), f"{method} {url}: HTTP {status}, expected a JSON object")
    return status, body


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def search_hits(body: dict[str, object], *, allow_empty: bool = False) -> list[dict]:
    data = body.get("data")
    check(isinstance(data, list), "search response must contain a data array")
    check((0 if allow_empty else 1) <= len(data) <= 100, "invalid search result count")
    check(all(
        isinstance(hit, dict)
        and isinstance(hit.get("id"), str) and bool(hit["id"])
        and isinstance(hit.get("content"), str) and bool(hit["content"])
        for hit in data
    ), "search items require nonempty string id and content")
    check(len({hit["id"] for hit in data}) == len(data), "duplicate search result IDs")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("AGENTMEM_API_KEY"),
                        help="Memory System Key (default: AGENTMEM_API_KEY)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="timeout per HTTP request in seconds (default: 600)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if args.api_key is not None and (
        not args.api_key or not args.api_key.isascii() or not args.api_key.isprintable()
        or any(char.isspace() for char in args.api_key)
    ):
        parser.error("--api-key / AGENTMEM_API_KEY must be nonempty printable ASCII without whitespace")
    base_url = args.base_url.rstrip("/")
    run_id = f"local-{uuid4().hex}"
    user_a, user_b = f"{run_id}:user-a", f"{run_id}:user-b"
    completed = []
    print(f"Smoke run: {run_id}", file=sys.stderr, flush=True)

    def passed(name: str) -> None:
        completed.append(name)
        print(f"[passed] {name}", file=sys.stderr, flush=True)

    def call(path: str, payload: dict | None = None, *, expected: int = 200) -> dict:
        status, body = request_json(
            "GET" if payload is None else "POST", base_url + path, payload,
            args.api_key if path != "/health" else None, args.timeout,
        )
        check(status == expected, f"{path}: expected HTTP {expected}, got {status}: {json.dumps(body, ensure_ascii=False)}")
        return body

    health = call("/health")
    check(health.get("status") == "ok", "health endpoint failed")
    passed("health")

    service = call("/status")
    channels = service.get("channels")
    check(isinstance(channels, dict) and channels.get("chunk") is True,
          "smoke test requires the chunk channel to be enabled")
    check(service.get("aml_search_content_format") == "native-excerpt-v1",
          "unsupported AML search content format")
    passed("service_status")

    def add(payload: dict) -> dict:
        body = call("/v1/memories/add", payload)
        check(body.get("success") is True, "add did not succeed")
        for field in ("request_id", "user_id", "session_id"):
            check(body.get(field) == payload[field], f"add response did not echo {field}")
        states = body.get("channels")
        check(isinstance(states, dict), "add response is missing channel states")
        for channel in ("chunk", "fact"):
            state = states.get(channel)
            expected = "succeeded" if channels.get(channel) else "disabled"
            check(isinstance(state, dict) and state.get("status") == expected,
                  f"{channel} did not finish as {expected}: {state}")
        check(body.get("complete") is True, f"add incomplete: {states}")
        return body

    payload_a = {
        "request_id": f"{run_id}:chunk-a", "user_id": user_a,
        "session_id": f"{run_id}:session-a",
        "messages": [
            {"role": "user", "speaker": "Ada", "timestamp": 1704067200000,
             "content": "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."},
            {"role": "assistant", "timestamp": 1704067201000,
             "content": "I will remember that historical fact."},
        ],
    }
    add(payload_a)
    passed("add_complete")

    query_a = {"query": "Who wrote notes about the Analytical Engine?", "user_id": user_a, "top_k": 100}
    original = search_hits(call("/search", query_a))
    check(any("Ada Lovelace" in hit["content"] for hit in original), "expected evidence was not retrieved")
    passed("local_search")

    add(payload_a)
    repeated = search_hits(call("/search", query_a))
    # Attempt counters change on replay. Persisted evidence IDs and text must not.
    check({hit["id"]: hit["content"] for hit in repeated}
          == {hit["id"]: hit["content"] for hit in original},
          "idempotent add changed or duplicated stored evidence")
    passed("idempotent_add")

    conflicting = {**payload_a, "messages": [{"role": "user", "content": "different payload"}]}
    call("/v1/memories/add", conflicting, expected=409)
    passed("request_conflict")

    state = call(f"/requests/{payload_a['request_id']}")
    check(state.get("complete") is True and state.get("processing") is False,
          f"request did not finish: {state}")
    for field in ("request_id", "user_id", "session_id"):
        check(state.get(field) == payload_a[field], f"request status did not echo {field}")
    passed("request_status")

    formal = search_hits(call("/v1/memories/search", query_a))
    check([hit["id"] for hit in formal] == [hit["id"] for hit in repeated],
          "formal and local search returned different evidence or ordering")
    for rank, (native, hit) in enumerate(zip(repeated, formal), 1):
        excerpt = json.loads(hit["content"])
        check(isinstance(excerpt, dict), "formal search content must encode a JSON object")
        check(excerpt.get("excerpt") == rank and excerpt.get("text") == native["content"],
              "formal search changed the evidence text or excerpt rank")
        for field in ("created_at", "type"):
            check(excerpt.get(field) == native.get(field), f"formal search changed {field}")
    passed("aml_search_context")

    marker = f"PRIVATE_MARKER_{uuid4().hex}"
    add({"request_id": f"{run_id}:chunk-b", "user_id": user_b,
         "session_id": f"{run_id}:session-b",
         "messages": [{"role": "user", "content": f"{marker} belongs only to user B."}]})
    for path in ("/search", "/v1/memories/search"):
        private_query = {"query": marker, "user_id": user_b, "top_k": 100}
        private = search_hits(call(path, private_query))
        check(any(marker in hit["content"] for hit in private), "private test evidence was not stored")
        isolated = search_hits(call(path, {**private_query, "user_id": user_a}), allow_empty=True)
        check(all(marker not in hit["content"] for hit in isolated), "cross-user memory leak detected")
    passed("user_isolation")

    empty_query = {"query": "anything", "user_id": f"{run_id}:empty", "top_k": 100}
    for path in ("/search", "/v1/memories/search"):
        check(call(path, empty_query) == {"data": []}, "empty user search must return an empty data array")
    passed("empty_user")

    print(json.dumps({"status": "passed", "run_id": run_id, "checks": len(completed),
                      "passed_checks": completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2),
              file=sys.stderr)
        sys.exit(1)
