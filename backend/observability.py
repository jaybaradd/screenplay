from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from backend.config import settings


class _NullObservation:
    def update(self, **_: Any) -> None:
        return None


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

    @contextmanager
    def span(self, name: str, *, project_id: str, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled or self.client is None:
            yield _NullObservation()
            return
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=project_id, metadata=metadata or {}):
            with self.client.start_as_current_observation(as_type="span", name=name) as observation:
                yield observation

    @contextmanager
    def generation(
        self, name: str, *, project_id: str, model: str, input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        captured_input = input_data if settings.langfuse_capture_content else {"input_hash_only": True}
        if not self.enabled or self.client is None:
            yield _NullObservation()
            return
        from langfuse import propagate_attributes

        trace_metadata = {
            "capture_content": str(settings.langfuse_capture_content).lower(),
            **{key: str(value) for key, value in (metadata or {}).items() if value is not None},
        }
        with propagate_attributes(session_id=project_id, metadata=trace_metadata):
            with self.client.start_as_current_observation(
                as_type="generation", name=name, model=model, input=captured_input
            ) as observation:
                yield observation

    def score(self, name: str, value: float, *, project_id: str, comment: str | None = None) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.create_score(name=name, value=value, session_id=project_id, comment=comment)
        except Exception:
            # Observability must never make the application unavailable.
            return


observer = Observer()
