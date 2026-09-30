from __future__ import annotations

from types import SimpleNamespace

from backend.providers import GeminiProvider, gemini_json_schema
from backend.schemas import ContinuityEventExtraction, CulturalBrief, CulturalClaim, SourceScreenplay


def brief() -> CulturalBrief:
    return CulturalBrief(
        culture_id="maidani_mewari", profile_version="1.0.0", profile_hash="test-profile",
        culture="Maidani Mewari", locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
        claims=[CulturalClaim(
            id="claim-1", claim="Test claim", scope="test", time_period="test",
            source_url="https://example.test/source", origin="grounded_research",
            confidence="low", layers=["cultural_precision"],
        )],
        constraints=[], open_questions=[], research_summary="Test",
    )


def test_gemini_structured_retries_twice_then_recovers(monkeypatch):
    calls = {"count": 0}

    def generate_content(**_):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("malformed response")
        return SimpleNamespace(parsed=brief(), text="")

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.text_model = "test-model"
    provider.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr("backend.providers.time.sleep", lambda _: None)
    result = provider.structured("prompt", CulturalBrief)
    assert result.culture == "Maidani Mewari"
    assert calls["count"] == 3


def test_gemini_structured_surfaces_recoverable_error_after_three_failures(monkeypatch):
    calls = {"count": 0}

    def generate_content(**_):
        calls["count"] += 1
        raise ValueError("malformed response")

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.text_model = "test-model"
    provider.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr("backend.providers.time.sleep", lambda _: None)
    try:
        provider.structured("prompt", CulturalBrief)
    except RuntimeError as error:
        assert "after 3 attempts" in str(error)
    else:
        raise AssertionError("Expected a recoverable structured-output error")
    assert calls["count"] == 3


def test_gemini_schema_uses_json_schema_path_without_unsupported_defaults():
    schema = gemini_json_schema(SourceScreenplay)
    encoded = str(schema)
    assert "$defs" in schema
    assert "properties" in schema
    assert "default" not in encoded
    assert "minLength" not in encoded
    # Strict Pydantic objects remain strict where Gemini JSON Schema supports it.
    assert schema["additionalProperties"] is False


def test_continuity_repair_schema_is_gemini_compatible():
    schema = gemini_json_schema(ContinuityEventExtraction)
    encoded = str(schema)
    assert "ContinuityEvent" in encoded
    assert "ProductionElement" in encoded
    assert "default" not in encoded
    assert "minLength" not in encoded
