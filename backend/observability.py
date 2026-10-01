from __future__ import annotations

import json
import inspect
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from backend.config import settings


_SECRET_KEY = re.compile(r"(?:api[-_]?key|secret|token|authorization|credential|password)", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def _redact_url(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            split = urlsplit(raw)
            query = [
                (key, "[REDACTED]" if _SECRET_KEY.search(key) else item)
                for key, item in parse_qsl(split.query, keep_blank_values=True)
            ]
            return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))
        except Exception:
            return raw

    return _URL.sub(replace, value)


def preview_text(value: str, limit: int | None = None) -> dict[str, Any]:
    limit = max(limit or settings.langfuse_preview_chars, 200)
    value = _redact_url(value)
    if len(value) <= limit:
        return {"text": value, "characters": len(value), "truncated": False}
    tail = min(2000, limit // 4)
    head = limit - tail
    omitted = len(value) - limit
    rendered = f"{value[:head]}\n\n[… {omitted} characters omitted …]\n\n{value[-tail:]}"
    return {"text": rendered, "characters": len(value), "truncated": True, "omitted_characters": omitted}


def preview_value(value: Any, *, limit: int | None = None, key: str | None = None) -> Any:
    """Build a useful, bounded and secret-redacted Langfuse payload."""
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        result = preview_text(value, limit)
        return result["text"] if not result["truncated"] else result
    if isinstance(value, dict):
        return {str(item_key): preview_value(item, limit=limit, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        maximum = 100
        rendered = [preview_value(item, limit=limit) for item in value[:maximum]]
        if len(value) > maximum:
            rendered.append({"omitted_items": len(value) - maximum})
        return rendered
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary": "omitted", "bytes": len(value)}
    return value


def captured_value(value: Any) -> Any:
    mode = settings.langfuse_content_mode
    if mode == "full":
        return preview_value(value, limit=max(settings.langfuse_preview_chars, 100_000))
    if mode == "metadata":
        payload = json.dumps(value, ensure_ascii=False, default=str)
        return {"content": "not captured", "characters": len(payload)}
    return preview_value(value)


class _NullObservation:
    id: str | None = None
    trace_id: str | None = None

    def update(self, **_: Any) -> None:
        return None


@dataclass
class _InvocationStats:
    invocation_id: str
    started: float = field(default_factory=time.perf_counter)
    model_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    outcome: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self, *, usage: dict[str, int] | None = None, cost: float | None = None,
        failed: bool = False, retry: bool = False, cache_hit: bool = False,
    ) -> None:
        with self._lock:
            if cache_hit:
                self.cache_hits += 1
                return
            self.model_calls += 1
            self.failures += int(failed)
            self.retries += int(retry)
            usage = usage or {}
            self.input_tokens += usage.get("input", 0)
            self.output_tokens += usage.get("output", 0)
            self.reasoning_tokens += usage.get("reasoning", 0)
            self.total_tokens += usage.get("total", 0)
            self.estimated_cost_usd += cost or 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "failed_model_calls": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 9),
            "latency_ms": int((time.perf_counter() - self.started) * 1000),
        }


_invocation: ContextVar[_InvocationStats | None] = ContextVar("observability_invocation", default=None)


class Observer:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self.client = None
        if self.enabled:
            try:
                from langfuse import get_client

                self.client = get_client()
            except Exception:
                self.enabled = False

    @property
    def invocation_id(self) -> str | None:
        current = _invocation.get()
        return current.invocation_id if current else None

    @contextmanager
    def command(
        self, name: str, *, project_id: str | None, input_data: Any = None,
        metadata: dict[str, Any] | None = None, tags: list[str] | None = None,
    ) -> Iterator[Any]:
        stats = _InvocationStats(str(uuid.uuid4()))
        token = _invocation.set(stats)
        root: Any = _NullObservation()
        enriched = {"invocation_id": stats.invocation_id, **(metadata or {})}
        try:
            if not self.enabled or self.client is None:
                yield root
                return
            from langfuse import propagate_attributes

            attributes: dict[str, Any] = {
                "trace_name": name,
                "metadata": enriched,
                "tags": tags or [],
                "version": settings.langfuse_release,
            }
            if "environment" in inspect.signature(propagate_attributes).parameters:
                attributes["environment"] = settings.langfuse_environment
            if project_id:
                attributes["session_id"] = project_id
            with propagate_attributes(**attributes):
                with self.client.start_as_current_observation(
                    as_type="span", name=name, input=captured_value(input_data), metadata=enriched,
                ) as root:
                    try:
                        yield root
                    except Exception as error:
                        self.safe_update(root, level="ERROR", status_message=str(error))
                        stats.outcome = {"result": "failed", "error_type": type(error).__name__}
                        raise
                    finally:
                        summary = {**stats.summary(), **stats.outcome}
                        self.safe_update(root, metadata={**enriched, **summary}, output=summary)
        finally:
            _invocation.reset(token)

    @contextmanager
    def span(
        self, name: str, *, project_id: str, metadata: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> Iterator[Any]:
        if not self.enabled or self.client is None:
            yield _NullObservation()
            return
        try:
            has_parent = bool(self.client.get_current_trace_id())
        except Exception:
            has_parent = False
        if has_parent:
            with self.client.start_as_current_observation(
                as_type="span", name=name, input=captured_value(input_data), metadata=metadata or {},
            ) as observation:
                yield observation
            return
        from langfuse import propagate_attributes

        attributes: dict[str, Any] = {
            "trace_name": name, "session_id": project_id, "metadata": metadata or {},
            "tags": [f"environment:{settings.langfuse_environment}"], "version": settings.langfuse_release,
        }
        if "environment" in inspect.signature(propagate_attributes).parameters:
            attributes["environment"] = settings.langfuse_environment
        with propagate_attributes(**attributes):
            with self.client.start_as_current_observation(
                as_type="span", name=name, input=captured_value(input_data), metadata=metadata or {},
            ) as observation:
                yield observation

    @contextmanager
    def generation(
        self, name: str, *, project_id: str, model: str, input_data: Any,
        metadata: dict[str, Any] | None = None, model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled or self.client is None:
            yield _NullObservation()
            return
        with self.client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=captured_value(input_data),
            metadata=metadata or {}, model_parameters=model_parameters or {},
        ) as observation:
            yield observation

    def safe_update(self, observation: Any, **values: Any) -> None:
        try:
            observation.update(**values)
        except Exception:
            return

    def identifiers(self, observation: Any) -> tuple[str | None, str | None]:
        observation_id = getattr(observation, "id", None) or getattr(observation, "observation_id", None)
        trace_id = getattr(observation, "trace_id", None)
        if self.client is not None:
            try:
                trace_id = trace_id or self.client.get_current_trace_id()
                observation_id = observation_id or self.client.get_current_observation_id()
            except Exception:
                pass
        return trace_id, observation_id

    def record_model_call(
        self, *, usage: dict[str, int] | None = None, cost: float | None = None,
        failed: bool = False, retry: bool = False, cache_hit: bool = False,
    ) -> None:
        current = _invocation.get()
        if current:
            current.record(usage=usage, cost=cost, failed=failed, retry=retry, cache_hit=cache_hit)

    def set_command_result(self, **values: Any) -> None:
        current = _invocation.get()
        if current:
            current.outcome.update(values)

    def score(self, name: str, value: float, *, project_id: str, comment: str | None = None) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            trace_id = self.client.get_current_trace_id()
            observation_id = self.client.get_current_observation_id()
            if trace_id:
                self.client.create_score(
                    name=name, value=value, trace_id=trace_id,
                    observation_id=observation_id, comment=comment,
                )
        except Exception:
            return

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            try:
                self.client.flush()
            except Exception:
                return

    def health(self) -> dict[str, Any]:
        from backend.telemetry import gemini_pricing

        return {
            "enabled": self.enabled,
            "sdk": "langfuse" if self.enabled else "noop",
            "capture_mode": settings.langfuse_content_mode,
            "environment": settings.langfuse_environment,
            "pricing_catalog": gemini_pricing.version,
            "pricing_tier": settings.gemini_pricing_tier,
        }


observer = Observer()
