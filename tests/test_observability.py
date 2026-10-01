from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.observability import Observer, preview_text, preview_value
from backend.providers import AIService
from backend.schemas import ProjectCreate
from backend.storage import Repository
from backend.telemetry import (
    GeminiPricing, ModelOperation, ProviderAttemptError, ProviderResult, ProviderTelemetry, ProviderUsage,
)


class DummyResult(BaseModel):
    value: str


class RetryingProvider:
    name = "google"
    text_model = "gemini-3.8-flash"
    image_model = "gemini-3.1-flash-image"

    def __init__(self):
        self.calls = 0

    def structured(self, prompt, schema):
        self.calls += 1
        if self.calls < 3:
            raise ProviderAttemptError(
                "malformed JSON",
                ProviderTelemetry(usage=ProviderUsage(prompt_tokens=10, total_tokens=10), provider_latency_ms=4),
            )
        return ProviderResult(
            value=DummyResult(value="recovered"),
            telemetry=ProviderTelemetry(
                usage=ProviderUsage(prompt_tokens=20, output_tokens=5, thinking_tokens=2, total_tokens=27),
                model_version="gemini-test", response_id="response-1", finish_reason="STOP",
                provider_latency_ms=8, parse_latency_ms=1,
            ),
        )


def make_project(repository: Repository) -> dict:
    return repository.create_project(ProjectCreate(
        title="Telemetry", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    ))


def test_operation_names_are_stable_and_uuid_is_metadata():
    asset_id = "0649d65d-b656-5274-ac0d-0f492bbb2a2b"
    operation = ModelOperation.from_legacy(f"verify-visual-{asset_id}")
    assert operation.name == "verify-visual"
    assert operation.target_id == asset_id
    assert asset_id not in operation.name


def test_command_and_generation_use_stable_names_and_model(monkeypatch):
    records = []

    class FakeObservation:
        id = "observation-1"
        trace_id = "trace-1"

        def __init__(self, payload):
            self.payload = payload
            self.updates = []

        def update(self, **values):
            self.updates.append(values)

    class FakeClient:
        active = False

        @contextmanager
        def start_as_current_observation(self, **payload):
            observation = FakeObservation(payload)
            records.append(observation)
            before = self.active
            self.active = True
            try:
                yield observation
            finally:
                self.active = before

        def get_current_trace_id(self):
            return "trace-1" if self.active else None

        def get_current_observation_id(self):
            return "observation-1" if self.active else None

    @contextmanager
    def propagate_attributes(**_):
        yield

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(propagate_attributes=propagate_attributes))
    observer = Observer.__new__(Observer)
    observer.enabled = True
    observer.client = FakeClient()
    with observer.command("asset.generate", project_id="project-1"):
        with observer.generation(
            "verify-visual", project_id="project-1", model="gemini-3.8-flash",
            input_data={"asset_id": "dynamic-id"}, metadata={"asset_id": "dynamic-id"},
        ):
            observer.record_model_call(usage={"input": 10, "output": 2, "total": 12}, cost=0.001)
        observer.set_command_result(result="success")
    assert [item.payload["name"] for item in records] == ["asset.generate", "verify-visual"]
    assert records[1].payload["model"] == "gemini-3.8-flash"
    assert "dynamic-id" not in records[1].payload["name"]
    assert records[0].updates[-1]["output"]["total_tokens"] == 12


def test_preview_preserves_head_tail_and_redacts_credentials():
    text = "H" * 180 + "MIDDLE" * 100 + "T" * 180
    preview = preview_text(text, 400)
    assert preview["truncated"] is True
    assert preview["text"].startswith("H")
    assert preview["text"].endswith("T" * 100)
    assert "characters omitted" in preview["text"]
    rendered = preview_value({
        "api_key": "should-never-appear",
        "url": "https://example.test/path?token=abc&safe=yes",
        "bytes": b"private-image-data",
    })
    assert rendered["api_key"] == "[REDACTED]"
    assert "token=%5BREDACTED%5D" in rendered["url"]
    assert rendered["bytes"]["binary"] == "omitted"


def test_text_and_image_cost_estimates_are_versioned():
    pricing = GeminiPricing()
    text = pricing.estimate(
        "gemini-3.8-flash",
        ProviderUsage(prompt_tokens=1_000_000, output_tokens=100_000, thinking_tokens=10_000),
        on_date=date(2026, 10, 1),
    )
    assert text.details["input"] == pytest.approx(0.75)
    assert text.details["output"] == pytest.approx(0.375)
    assert text.details["reasoning"] == pytest.approx(0.0375)
    assert text.catalog_version == "gemini-2026-10-01"

    image = pricing.estimate(
        "gemini-3.1-flash-image", ProviderUsage(prompt_tokens=1000),
        on_date=date(2026, 10, 1), image_dimensions=(1024, 1024),
    )
    assert image.details["image_output"] == pytest.approx(0.0672)
    assert image.calculation_basis == "resolution_fallback"


def test_ai_service_records_each_retry_and_reuses_successful_cache(tmp_path: Path, monkeypatch):
    repository = Repository(tmp_path / "app.sqlite")
    project = make_project(repository)
    service = AIService(repository)
    provider = RetryingProvider()
    service.provider = provider
    monkeypatch.setattr("backend.providers.time.sleep", lambda _: None)

    result = service.cached_structured(project["id"], "test-operation", "readable prompt", DummyResult)
    assert result.value == "recovered"
    assert provider.calls == 3
    runs = repository.list_model_runs(project["id"])
    assert len(runs) == 1
    assert runs[0]["operation"] == "test-operation"
    assert runs[0]["attempt_count"] == 3
    assert len(runs[0]["attempts"]) == 3
    assert [item["status"] for item in runs[0]["attempts"]] == ["failed", "failed", "success"]
    assert runs[0]["total_tokens"] == 47
    assert runs[0]["estimated_cost_usd"] is not None

    cached = service.cached_structured(project["id"], "test-operation", "readable prompt", DummyResult)
    assert cached.value == "recovered"
    assert provider.calls == 3
    assert repository.list_model_runs(project["id"])[0]["cache_hit_count"] == 1
    connection = sqlite3.connect(repository.path)
    local_input = connection.execute("SELECT input_json FROM model_runs").fetchone()[0]
    connection.close()
    assert "readable prompt" in local_input


def test_v2_database_migrates_without_losing_model_runs(tmp_path: Path):
    database = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES ('schema_version','2');
        CREATE TABLE projects (id TEXT PRIMARY KEY);
        CREATE TABLE model_runs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, operation TEXT NOT NULL,
            provider TEXT NOT NULL, model TEXT NOT NULL, prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL, cache_key TEXT NOT NULL UNIQUE, culture_id TEXT,
            profile_hash TEXT, brief_revision_id TEXT, output_json TEXT, status TEXT NOT NULL,
            latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, error TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO projects VALUES ('project-1');
        INSERT INTO model_runs VALUES (
            'run-1','project-1','verify-visual-old-id','google','gemini-3.8-flash',
            'prompt','input','cache',NULL,NULL,NULL,'{}','success',100,NULL,NULL,NULL,'2026-01-01'
        );
    """)
    connection.commit()
    connection.close()

    repository = Repository(database)
    connection = sqlite3.connect(database)
    version = connection.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0]
    columns = {row[1] for row in connection.execute("PRAGMA table_info(model_runs)")}
    preserved = connection.execute("SELECT id,operation FROM model_runs").fetchone()
    connection.close()
    assert version == "3"
    assert {"total_tokens", "cost_json", "attempt_count", "langfuse_trace_id"} <= columns
    assert preserved == ("run-1", "verify-visual-old-id")
