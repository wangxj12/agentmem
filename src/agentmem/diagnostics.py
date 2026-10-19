"""Request-correlated diagnostics that avoid additional model calls or writes."""

import contextvars
import functools
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
_current = contextvars.ContextVar("agentmem_diagnostics", default=None)
_event_observer = contextvars.ContextVar("agentmem_event_observer", default=None)


@contextmanager
def observe_events(callback):
    """Observe existing events without changing channel logic or requiring log files."""
    token = _event_observer.set(callback)
    try:
        yield
    finally:
        _event_observer.reset(token)


class Diagnostics:
    def __init__(self, path=None, *, include_content=False, secrets=()):
        self.path = Path(path) if path else None
        self.include_content = include_content
        self.secrets = tuple(str(s) for s in secrets if s)
        self.lock = threading.Lock()

    def scrub(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    @contextmanager
    def scope(self, operation, **identity):
        context = {"operation": operation, "trace_id": uuid.uuid4().hex, **identity}
        token = _current.set((self, context))
        try:
            yield context
        finally:
            _current.reset(token)

    def write(self, row):
        if self.path is None:
            return
        try:
            encoded = self.scrub(
                json.dumps(row, ensure_ascii=False, allow_nan=False, default=json_value)
            )
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
        except Exception:  # noqa: BLE001 -- diagnostics must never fail a business operation
            # Diagnostic failures must never turn a successful memory write into a failure.
            logger.warning(
                "Diagnostic output unavailable",
                extra={"event": "diagnostic_write_failed"},
            )


def scrub(text):
    current = _current.get()
    return current[0].scrub(text) if current else text


def emit(event, *, content=None, **fields):
    observer = _event_observer.get()
    if observer is not None:
        observer(event, {**fields, '_content': content})
    current = _current.get()
    if current is None:
        return
    sink, identity = current
    if sink.path is None:
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        **identity,
        "event": event,
        **fields,
    }
    logger.info(event, extra={"diagnostic": row})
    if sink.path is not None:
        if sink.include_content and content is not None:
            row = {**row, "content": content}
        sink.write(row)


def error_fields(error):
    cause = error
    seen = set()
    while cause.__cause__ is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause = cause.__cause__
    return {
        "error_type": type(cause).__name__,
        "http_status": getattr(cause, "status_code", None)
        or getattr(getattr(cause, "response", None), "status_code", None),
    }


@contextmanager
def stage(name, **fields):
    start = time.perf_counter()
    metrics = {}
    emit("stage_started", stage=name, **fields)
    try:
        yield metrics
    except Exception as error:
        emit(
            "stage_finished",
            stage=name,
            status="error",
            elapsed_seconds=time.perf_counter() - start,
            **fields,
            **metrics,
            **error_fields(error),
        )
        raise
    else:
        emit(
            "stage_finished",
            stage=name,
            status="ok",
            elapsed_seconds=time.perf_counter() - start,
            **fields,
            **metrics,
        )


def observed(name, *, channel=None):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            kind = channel or getattr(self, "channel", None)
            with stage(name, channel=kind):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate


def json_value(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")
