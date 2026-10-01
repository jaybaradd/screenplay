from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ModelOperation:
    """Stable human-facing operation plus high-cardinality target metadata.

    ``cache_scope`` intentionally retains the pre-observability operation string so
    existing successful cache entries remain reusable.
    """

    name: str
    modality: str = "text"
    cache_scope: str = ""
    target_type: str | None = None
    target_id: str | None = None
    scene_number: int | None = None
    layer: str | None = None
    schema_name: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None and key != "cache_scope"}

    @classmethod
    def from_legacy(
        cls, operation: str, *, schema_name: str | None = None, modality: str = "text",
    ) -> "ModelOperation":
        name = operation
        target_type: str | None = None
        target_id: str | None = None
        scene_number: int | None = None
        layer: str | None = None

        if operation.startswith("plan-"):
            name, layer = "plan-layer", operation.removeprefix("plan-")
        elif match := re.fullmatch(r"adapt-scene-(\d+)-attempt-(\d+)", operation):
            scene_number = int(match.group(1))
            name = "adapt-scene" if match.group(2) == "1" else "repair-adapted-scene"
            target_type, target_id = "scene", str(scene_number)
        elif operation.startswith("repair-continuity-"):
            name = "repair-continuity"
        elif operation.startswith("correct-block-"):
            name, target_type = "correct-screenplay-block", "block"
            target_id = operation.removeprefix("correct-block-")
        elif operation.startswith("build-visual-manifest"):
            name = "build-visual-manifest"
        elif operation.startswith("generate-image-"):
            name, modality, target_type = "generate-image", "image", "asset"
            target_id = operation.removeprefix("generate-image-")
        elif operation.startswith("verify-visual-"):
            name, modality, target_type = "verify-visual", "vision", "asset"
            target_id = operation.removeprefix("verify-visual-")

        return cls(
            name=name, modality=modality, cache_scope=operation, target_type=target_type,
            target_id=target_id, scene_number=scene_number, layer=layer, schema_name=schema_name,
        )


@dataclass
class ProviderUsage:
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    tool_use_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    input_modalities: dict[str, int] = field(default_factory=dict)
    output_modalities: dict[str, int] = field(default_factory=dict)
    grounding_queries: int = 0

    @property
    def image_output_tokens(self) -> int | None:
        value = sum(count for modality, count in self.output_modalities.items() if modality.lower() == "image")
        return value or None

    def langfuse_details(self) -> dict[str, int]:
        values = {
            "input": self.prompt_tokens,
            "output": self.output_tokens,
            "reasoning": self.thinking_tokens,
            "tool_use": self.tool_use_tokens,
            "cache_read": self.cached_tokens,
            "total": self.total_tokens,
            "image_output": self.image_output_tokens,
            "grounding_queries": self.grounding_queries or None,
        }
        return {key: int(value) for key, value in values.items() if value is not None}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderTelemetry:
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    model_version: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    provider_latency_ms: int | None = None
    parse_latency_ms: int | None = None
    io_latency_ms: int | None = None


@dataclass
class ProviderResult(Generic[T]):
    value: T
    telemetry: ProviderTelemetry = field(default_factory=ProviderTelemetry)


class ProviderAttemptError(RuntimeError):
    def __init__(self, message: str, telemetry: ProviderTelemetry | None = None):
        super().__init__(message)
        self.telemetry = telemetry or ProviderTelemetry()


@dataclass
class CostEstimate:
    details: dict[str, float] = field(default_factory=dict)
    total: float | None = None
    catalog_version: str | None = None
    cost_basis: str = "public_list_price_estimate"
    calculation_basis: str = "provider_usage"
    free_allowance_applied: bool = False


class GeminiPricing:
    def __init__(self, path: Path | None = None):
        path = path or Path(__file__).with_name("pricing_catalog.json")
        self.catalog = json.loads(path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return self.catalog["catalog_version"]

    def _rates(self, model: str, on_date: date) -> dict[str, Any] | None:
        for period in self.catalog.get("models", {}).get(model, []):
            start = date.fromisoformat(period["effective_from"])
            end = date.fromisoformat(period["effective_to"]) if period.get("effective_to") else None
            if start <= on_date and (end is None or on_date <= end):
                return period
        return None

    def estimate(
        self, model: str, usage: ProviderUsage, *, on_date: date | None = None,
        image_dimensions: tuple[int, int] | None = None,
    ) -> CostEstimate:
        rates = self._rates(model, on_date or date.today())
        if not rates:
            return CostEstimate(catalog_version=self.version, calculation_basis="unknown_model")

        prompt = usage.prompt_tokens or 0
        cached = min(usage.cached_tokens or 0, prompt)
        tool = usage.tool_use_tokens or 0
        image_tokens = usage.image_output_tokens
        output = usage.output_tokens or 0
        basis = "provider_usage"
        if image_dimensions and not image_tokens and rates.get("image_token_fallbacks"):
            longest = max(image_dimensions)
            buckets = sorted((int(size), tokens) for size, tokens in rates["image_token_fallbacks"].items())
            image_tokens = next((tokens for size, tokens in buckets if longest <= size), buckets[-1][1])
            basis = "resolution_fallback"
        image_tokens = image_tokens or 0
        text_output = max(output - image_tokens, 0)

        per_million = 1_000_000
        details = {
            "input": ((prompt - cached + tool) * rates.get("input_per_million", 0)) / per_million,
            "output": (text_output * rates.get("output_per_million", 0)) / per_million,
            "reasoning": ((usage.thinking_tokens or 0) * rates.get("thinking_per_million", 0)) / per_million,
            "cache_read": (cached * rates.get("cache_read_per_million", 0)) / per_million,
            "image_output": (image_tokens * rates.get("image_output_per_million", 0)) / per_million,
            "search_grounding": (
                usage.grounding_queries * self.catalog.get("grounding", {}).get("google_search_per_query", 0)
            ),
        }
        details = {key: round(value, 9) for key, value in details.items() if value}
        return CostEstimate(
            details=details, total=round(sum(details.values()), 9), catalog_version=self.version,
            calculation_basis=basis,
            free_allowance_applied=self.catalog.get("grounding", {}).get("free_allowance_applied", False),
        )


gemini_pricing = GeminiPricing()
