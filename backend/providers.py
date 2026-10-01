from __future__ import annotations

import hashlib
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from PIL import Image, ImageDraw
from pydantic import BaseModel

from backend.config import settings
from backend.observability import observer, preview_value
from backend.prompts import PROMPT_VERSION
from backend.schemas import (
    AdaptedBlock, AdaptedScene, AdaptedScreenplay, BlockType, CharacterRecord,
    ContentBlock, CulturalBrief, CulturalClaim, CulturalConstraint, LanguageGuide, LayerDecision, LayerPlan,
    ProductionElement, SceneRecord, SourceScreenplay, StoryContract,
    VisualVerification, utc_now,
)
from backend.storage import Repository, content_hash
from backend.telemetry import (
    ModelOperation, ProviderAttemptError, ProviderResult, ProviderTelemetry, ProviderUsage, gemini_pricing,
)


T = TypeVar("T", bound=BaseModel)
ImageReference: TypeAlias = tuple[str, Path]

_GEMINI_JSON_SCHEMA_KEYS = {
    "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum",
    "items", "prefixItems", "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf",
    "properties", "additionalProperties", "required", "propertyOrdering",
}


def gemini_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON-Schema subset accepted by Gemini structured output.

    Pydantic emits useful local-only keywords such as ``default``, ``minLength`` and
    ``maxLength``. Gemini rejects some of them. Property and definition names are
    preserved while unsupported schema keywords are removed; Pydantic validates the
    returned JSON again after generation, so application-side strictness remains intact.
    """

    def clean(value: Any, parent: str | None = None) -> Any:
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        if parent in {"properties", "$defs"}:
            return {key: clean(item) for key, item in value.items()}
        return {
            key: clean(item, key)
            for key, item in value.items()
            if key in _GEMINI_JSON_SCHEMA_KEYS
        }

    return clean(model.model_json_schema())


def parse_structured_response(response: Any, schema: type[T]) -> T:
    """Parse all structured-response shapes returned by google-genai.

    Depending on SDK/model behavior, ``parsed`` may be a model, a plain dict,
    or absent while JSON is present only in a candidate part.  ``response.text``
    may legitimately be ``None`` for an empty/safety-truncated response.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if parsed is not None:
        return schema.model_validate(parsed)

    candidates: list[str] = []
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        candidates.append(response_text)
    finish_reasons: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            finish_reasons.append(str(finish_reason))
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                candidates.append(part_text)
    last_error: Exception | None = None
    for candidate_text in candidates:
        cleaned = candidate_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        try:
            return schema.model_validate_json(cleaned)
        except Exception as error:
            last_error = error
    diagnostic = f" finish_reasons={finish_reasons}" if finish_reasons else ""
    if last_error:
        raise RuntimeError(f"Gemini returned malformed structured JSON:{diagnostic} {last_error}") from last_error
    raise RuntimeError(f"Gemini returned no structured text or parsed payload.{diagnostic}")


def stable_id(project_id: str, kind: str, key: str) -> str:
    return str(uuid.uuid5(uuid.UUID(project_id), f"{kind}:{key.strip().lower()}"))


class Provider(ABC):
    name: str
    text_model: str
    image_model: str

    @abstractmethod
    def structured(self, prompt: str, schema: type[T]) -> ProviderResult[T]: ...

    @abstractmethod
    def grounded_research(self, prompt: str) -> ProviderResult[str]: ...

    @abstractmethod
    def generate_image(
        self, prompt: str, output: Path, references: list[ImageReference] | None = None,
    ) -> ProviderResult[dict[str, Any]]: ...

    @abstractmethod
    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> ProviderResult[VisualVerification]: ...


def _modality_counts(items: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items or []:
        raw = getattr(item, "modality", None)
        name = getattr(raw, "value", raw)
        name = str(name or "unknown").split(".")[-1].lower()
        count = getattr(item, "token_count", None)
        if count is not None:
            result[name] = result.get(name, 0) + int(count)
    return result


def _response_telemetry(
    response: Any, *, provider_latency_ms: int | None = None, parse_latency_ms: int | None = None,
) -> ProviderTelemetry:
    metadata = getattr(response, "usage_metadata", None)
    usage = ProviderUsage(
        prompt_tokens=getattr(metadata, "prompt_token_count", None),
        output_tokens=getattr(metadata, "candidates_token_count", None),
        thinking_tokens=getattr(metadata, "thoughts_token_count", None),
        tool_use_tokens=getattr(metadata, "tool_use_prompt_token_count", None),
        cached_tokens=getattr(metadata, "cached_content_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
        input_modalities=_modality_counts(getattr(metadata, "prompt_tokens_details", None)),
        output_modalities=_modality_counts(getattr(metadata, "candidates_tokens_details", None)),
    )
    finish_reasons = [
        str(getattr(reason, "value", reason))
        for candidate in (getattr(response, "candidates", None) or [])
        if (reason := getattr(candidate, "finish_reason", None)) is not None
    ]
    search_queries: set[str] = set()
    for candidate in getattr(response, "candidates", None) or []:
        grounding = getattr(candidate, "grounding_metadata", None)
        search_queries.update(str(item) for item in (getattr(grounding, "web_search_queries", None) or []))
    usage.grounding_queries = len(search_queries)
    return ProviderTelemetry(
        usage=usage,
        model_version=getattr(response, "model_version", None),
        response_id=getattr(response, "response_id", None),
        finish_reason=", ".join(dict.fromkeys(finish_reasons)) or None,
        provider_latency_ms=provider_latency_ms,
        parse_latency_ms=parse_latency_ms,
    )


class GeminiProvider(Provider):
    name = "google"

    def __init__(self) -> None:
        from google import genai

        self.text_model = settings.text_model
        self.image_model = settings.image_model
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def structured(self, prompt: str, schema: type[T]) -> ProviderResult[T]:
        from google.genai import types

        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.text_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=gemini_json_schema(schema),
                ),
            )
        except Exception as error:
            raise ProviderAttemptError(
                str(error), ProviderTelemetry(provider_latency_ms=int((time.perf_counter() - started) * 1000)),
            ) from error
        provider_latency = int((time.perf_counter() - started) * 1000)
        parse_started = time.perf_counter()
        try:
            value = parse_structured_response(response, schema)
        except Exception as error:
            telemetry = _response_telemetry(
                response, provider_latency_ms=provider_latency,
                parse_latency_ms=int((time.perf_counter() - parse_started) * 1000),
            )
            raise ProviderAttemptError(str(error), telemetry) from error
        telemetry = _response_telemetry(
            response, provider_latency_ms=provider_latency,
            parse_latency_ms=int((time.perf_counter() - parse_started) * 1000),
        )
        return ProviderResult(value=value, telemetry=telemetry)

    def grounded_research(self, prompt: str) -> ProviderResult[str]:
        from google.genai import types

        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.text_model,
                contents=prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
            )
        except Exception as error:
            raise ProviderAttemptError(
                str(error), ProviderTelemetry(provider_latency_ms=int((time.perf_counter() - started) * 1000)),
            ) from error
        sources: list[str] = []
        for candidate in response.candidates or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if uri:
                    sources.append(f"- {title or 'Source'}: {uri}")
        value = f"{response.text or ''}\n\nSOURCES\n" + "\n".join(dict.fromkeys(sources))
        return ProviderResult(
            value=value,
            telemetry=_response_telemetry(response, provider_latency_ms=int((time.perf_counter() - started) * 1000)),
        )

    def generate_image(
        self, prompt: str, output: Path, references: list[ImageReference] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        from google.genai import types

        contents: list[Any] = [prompt]
        for label, path in references or []:
            contents.append(f"REFERENCE IMAGE — {label}")
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(model=self.image_model, contents=contents)
        except Exception as error:
            raise ProviderAttemptError(
                str(error), ProviderTelemetry(provider_latency_ms=int((time.perf_counter() - started) * 1000)),
            ) from error
        provider_latency = int((time.perf_counter() - started) * 1000)
        for part in response.parts or []:
            if getattr(part, "inline_data", None):
                write_started = time.perf_counter()
                output.write_bytes(part.inline_data.data)
                io_latency = int((time.perf_counter() - write_started) * 1000)
                telemetry = _response_telemetry(response, provider_latency_ms=provider_latency)
                telemetry.io_latency_ms = io_latency
                return ProviderResult(
                    value={"mime_type": getattr(part.inline_data, "mime_type", "image/png")},
                    telemetry=telemetry,
                )
        raise ProviderAttemptError(
            "Gemini returned no image data",
            _response_telemetry(response, provider_latency_ms=int((time.perf_counter() - started) * 1000)),
        )

    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> ProviderResult[VisualVerification]:
        from google.genai import types

        contents: list[Any] = [
            "Verify this generated production image against the approved specification and every labelled canonical "
            "reference. Check face, apparent age, body, grooming, costume, props, location, set geometry, spatial layout, composition "
            "and visual style. Treat illustration, cartoon, animation, anime, comic, painting, vector-art or obvious 3D-rendered "
            "output as a blocking visual_style failure whenever the specification requires photorealistic live action. "
            "When references are supplied, enforce their overall medium and realism while preserving the target identity. "
            "Return blocking issues for identity, costume, prop, location or required-style contradictions. Do not infer cultural facts "
            f"outside the specification. asset_id={asset_id}; dependency_hash={dependency_hash}; specification={prompt}",
            types.Part.from_bytes(data=image.read_bytes(), mime_type="image/png"),
        ]
        for label, path in references or []:
            contents.append(f"REFERENCE IMAGE — {label}")
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.text_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=gemini_json_schema(VisualVerification),
                    temperature=0,
                ),
            )
        except Exception as error:
            raise ProviderAttemptError(
                str(error), ProviderTelemetry(provider_latency_ms=int((time.perf_counter() - started) * 1000)),
            ) from error
        provider_latency = int((time.perf_counter() - started) * 1000)
        parse_started = time.perf_counter()
        try:
            result = parse_structured_response(response, VisualVerification)
        except Exception as error:
            telemetry = _response_telemetry(
                response, provider_latency_ms=provider_latency,
                parse_latency_ms=int((time.perf_counter() - parse_started) * 1000),
            )
            raise ProviderAttemptError(str(error), telemetry) from error
        if result.asset_id != asset_id or result.dependency_hash != dependency_hash:
            raise ProviderAttemptError(
                "Visual verifier returned mismatched asset identity",
                _response_telemetry(response, provider_latency_ms=provider_latency),
            )
        return ProviderResult(
            value=result,
            telemetry=_response_telemetry(
                response, provider_latency_ms=provider_latency,
                parse_latency_ms=int((time.perf_counter() - parse_started) * 1000),
            ),
        )


class MockProvider(Provider):
    name = "mock"
    text_model = "deterministic-mock-v1"
    image_model = "deterministic-mock-image-v1"

    def __init__(self) -> None:
        self.context: dict[str, Any] = {}

    def set_context(self, **values: Any) -> None:
        self.context.update(values)

    def structured(self, prompt: str, schema: type[T]) -> ProviderResult[T]:
        if schema is CulturalBrief:
            value = self._cultural_brief()
            return ProviderResult(value=value)  # type: ignore[arg-type,return-value]
        if schema is LayerPlan:
            match = re.search(r"layer=(\w+)", prompt)
            value = self._layer_plan(match.group(1) if match else "verbal")
            return ProviderResult(value=value)  # type: ignore[arg-type,return-value]
        raise NotImplementedError(f"Mock structured output not implemented for {schema.__name__}")

    def grounded_research(self, prompt: str) -> ProviderResult[str]:
        return ProviderResult(value=(
            "MOCK MODE: No web research was performed. The brief will contain only cautious workflow constraints, "
            "and must not be presented as verified cultural evidence. Configure GEMINI_API_KEY and AI_MODE=live for grounding."
        ))

    def generate_image(
        self, prompt: str, output: Path, references: list[ImageReference] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        digest = content_hash(prompt)
        colour = tuple(int(digest[index:index + 2], 16) for index in (0, 2, 4))
        image = Image.new("RGB", (1024, 1024), colour)
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 60, 964, 964), outline="white", width=6)
        draw.text((90, 100), "MVP MOCK VISUAL", fill="white")
        draw.text((90, 145), content_hash(prompt)[:16], fill="white")
        draw.multiline_text((90, 210), prompt[:700], fill="white", spacing=8)
        image.save(output, format="PNG")
        return ProviderResult(value={"mime_type": "image/png", "mock": True})

    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> ProviderResult[VisualVerification]:
        return ProviderResult(value=VisualVerification(
            asset_id=asset_id, dependency_hash=dependency_hash, passed=True, issues=[],
            summary="Mock verifier confirms only workflow plumbing, not visual or cultural accuracy.",
        ))

    def extract(self, project_id: str, title: str, text: str) -> SourceScreenplay:
        heading_pattern = re.compile(r"(?im)^(INT\.?/EXT\.?|I/E\.?|INT\.?|EXT\.?)\s+(.+)$")
        matches = list(heading_pattern.finditer(text))
        if not matches:
            matches = [re.match(r"", text)]  # type: ignore[list-item]
        scenes: list[SceneRecord] = []
        characters: dict[str, CharacterRecord] = {}
        elements: dict[str, ProductionElement] = {}
        for index, match in enumerate(matches):
            start = match.start() if match else 0
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            heading = lines[0] if match and lines else f"SCENE {index + 1}"
            int_ext = (match.group(1).replace(".", "") if match else "UNKNOWN").upper()
            location_time = match.group(2) if match else "UNSPECIFIED"
            location, _, time_label = location_time.partition(" - ")
            scene_id = stable_id(project_id, "scene", str(index + 1))
            location_id = stable_id(project_id, "location", location)
            elements.setdefault(location_id, ProductionElement(
                id=location_id, kind="location", name=location.title(), scene_ids=[]
            )).scene_ids.append(scene_id)
            blocks: list[ContentBlock] = []
            scene_characters: list[str] = []
            cursor = start
            line_index = 1 if match else 0
            while line_index < len(lines):
                line = lines[line_index]
                is_speaker = bool(re.fullmatch(r"[A-Z][A-Z0-9 .'-]{1,35}(?:\s*\([^)]*\))?", line))
                if is_speaker and line_index + 1 < len(lines):
                    speaker_name = re.sub(r"\s*\([^)]*\)$", "", line).strip().title()
                    character_id = stable_id(project_id, "character", speaker_name)
                    characters.setdefault(character_id, CharacterRecord(
                        id=character_id, name=speaker_name, aliases=[], role="Extracted speaker",
                        scene_entrances=[scene_id], emotional_state="Unspecified",
                    ))
                    if scene_id not in characters[character_id].scene_entrances:
                        characters[character_id].scene_entrances.append(scene_id)
                    scene_characters.append(character_id)
                    dialogue = lines[line_index + 1]
                    block_id = stable_id(project_id, "block", f"{index + 1}:{len(blocks)}")
                    blocks.append(ContentBlock(
                        id=block_id, type=BlockType.dialogue, text=dialogue,
                        speaker_id=character_id, speaker_label=speaker_name,
                        source_start=cursor, source_end=cursor + len(dialogue),
                    ))
                    cursor += len(line) + len(dialogue) + 2
                    line_index += 2
                else:
                    block_id = stable_id(project_id, "block", f"{index + 1}:{len(blocks)}")
                    blocks.append(ContentBlock(
                        id=block_id, type=BlockType.action, text=line,
                        source_start=cursor, source_end=cursor + len(line),
                    ))
                    cursor += len(line) + 1
                    line_index += 1
            scenes.append(SceneRecord(
                id=scene_id, number=index + 1, heading=heading, int_ext=int_ext,
                location_id=location_id, location=location.title(), sub_location="Not stated in source",
                time=time_label or "Not stated in source", day_or_date="Not stated in source",
                weather="Not stated in source", mood="Requires human review in mock mode",
                summary=" ".join(block.text for block in blocks[:2])[:240] or "Scene extracted from source.",
                dramatic_purpose="Preserve the source scene's causal and emotional function.",
                blocks=blocks, character_ids=list(dict.fromkeys(scene_characters)),
                production_element_ids=[location_id],
            ))
        contract = StoryContract(
            central_dramatic_purpose="Preserve the source conflict and its resolution.",
            relationship_invariants=["Do not change established relationships without explicit approval."],
            scene_purposes={scene.id: scene.dramatic_purpose for scene in scenes},
            plot_invariants=["Preserve scene order and causal events."],
            emotional_arc=["Preserve each character's source emotional progression."],
            prohibited_changes=["No new identity, faith, caste, occupation or motivation assumptions."],
        )
        return SourceScreenplay(
            title=title, detected_language="Undetermined (mock mode)", original_text=text,
            scenes=scenes, characters=list(characters.values()), production_elements=list(elements.values()),
            story_contract=contract,
            extraction_warnings=["Mock extraction is deterministic and must be reviewed before approval."],
        )

    def adapt(self, source: SourceScreenplay, output_script: str) -> AdaptedScreenplay:
        scenes: list[AdaptedScene] = []
        for scene in source.scenes:
            adapted_blocks = [AdaptedBlock(
                id=block.id, source_block_ids=[block.id], type=block.type, speaker_id=block.speaker_id,
                adapted_text=block.text,
                adaptation_layer_ids=["mock-preservation"], cultural_claim_ids=[],
                explanation="Mock mode preserves the source text; live Gemini mode performs the cultural rewrite.",
                confidence="low", changed_dimensions=[],
            ) for block in scene.blocks]
            scenes.append(AdaptedScene(
                id=scene.id, source_scene_id=scene.id, heading=scene.heading,
                summary=scene.summary, blocks=adapted_blocks,
            ))
        return AdaptedScreenplay(
            title=source.title, output_script=output_script, scenes=scenes,
            preservation_summary="Mock mode preserved all source blocks exactly for workflow verification.",
        )

    def _cultural_brief(self) -> CulturalBrief:
        context = self.context.get("culture_context", {})
        profile_constraints = [
            CulturalConstraint(
                id=item["id"], text=item["text"], layers=item["layers"],
                kind=item["kind"], origin="profile_policy", source_claim_ids=[],
            ) for item in context.get("constraints", [])
            if item.get("origin") == "profile_policy"
        ]
        return CulturalBrief(
            culture_id=context.get("culture_id", "mock-culture"),
            profile_version=context.get("profile_version", "0.0.0"),
            profile_hash=context.get("profile_hash", "mock-profile"),
            culture=context.get("display_name", "Mock culture"),
            locality=context.get("locality", "User-selected locality"),
            setting=context.get("setting", "unspecified"), period=context.get("period", "unspecified"),
            output_script=context.get("output_script", "unspecified"),
            claims=[CulturalClaim(
                id="mock-claim", claim="No cultural factual claim is asserted in mock mode.",
                scope="Workflow demonstration only", time_period="Not applicable", source_url="mock://no-live-grounding",
                origin="mock",
                confidence="low", layers=["cultural_precision"],
                uncertainty="Configure live Gemini grounding before presenting cultural output.",
                prohibited_extrapolations=["Do not treat mock content as cultural evidence."],
            )],
            language_guide=LanguageGuide(
                target_variety=context.get("target_variety", "Mock target variety"),
                writing_script=context.get("output_script", "unspecified"), features=[],
                register_rules=[], code_switching_rules=[],
                negative_constraints=["Mock mode cannot supply culturally verified dialogue."],
            ),
            constraints=profile_constraints,
            open_questions=["Live cultural grounding has not run."],
            research_summary="Mock mode validates engineering flow, not cultural accuracy.",
        )

    def _layer_plan(self, layer: str) -> LayerPlan:
        allowed = {"verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision"}
        if layer not in allowed:
            layer = "verbal"
        return LayerPlan(
            layer=layer, objective=f"Review {layer.replace('_', ' ')} adaptations without changing story invariants.",
            decisions=[LayerDecision(
                id=f"mock-{layer}-decision", source_observations=["Source records require human review."],
                proposed_changes=["No factual cultural change is applied in mock mode."],
                affected_scene_ids=[], affected_character_ids=[],
                preserved_invariants=["Scene order", "Relationships", "Emotional arc"],
                cultural_claim_ids=["mock-claim"], uncertainty=["Live grounding unavailable"],
                language_feature_ids=[],
                risks=["Mock output is not culturally validated"], do_not_change=["Core story logic"],
                negative_constraints=["Do not invent culture-specific detail"],
            )],
        )


class AIService:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.provider: Provider = GeminiProvider() if settings.ai_mode == "live" and settings.gemini_api_key else MockProvider()

    def _culture_metadata(self, project_id: str) -> dict[str, Any]:
        project = self.repository.get_project(project_id)
        brief = self.repository.latest_revision(project_id, "cultural_brief")
        return {
            "culture_id": project.get("culture_id"),
            "profile_hash": project.get("profile_hash"),
            "brief_revision_id": brief["id"] if brief else None,
        }

    def _asset_metadata(self, project_id: str, asset_id: str) -> dict[str, Any]:
        try:
            asset = self.repository.get_asset(asset_id)
        except KeyError:
            return {"asset_id": asset_id}
        metadata: dict[str, Any] = {
            "asset_id": asset_id, "asset_kind": asset.get("kind"),
            "canonical_id": asset.get("canonical_id"), "scene_id": asset.get("scene_id"),
        }
        if asset.get("scene_id"):
            extraction = self.repository.latest_revision(project_id, "extraction")
            scene = next(
                (item for item in (extraction or {}).get("payload", {}).get("scenes", []) if item["id"] == asset["scene_id"]),
                None,
            )
            if scene:
                metadata["scene_number"] = scene.get("number")
        return {key: value for key, value in metadata.items() if value is not None}

    @staticmethod
    def _reference_metadata(references: list[ImageReference] | None) -> list[dict[str, Any]]:
        results = []
        for label, path in references or []:
            item: dict[str, Any] = {
                "label": label, "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "filename": path.name,
            }
            try:
                with Image.open(path) as image:
                    item.update({"width": image.width, "height": image.height, "format": image.format})
            except Exception:
                item["dimensions"] = "unavailable"
            results.append(item)
        return results

    def _execute(
        self, *, project_id: str, operation: ModelOperation, model: str, prompt_hash: str,
        input_hash: str, cache_key: str, input_data: Any, invoke, output_builder,
        model_parameters: dict[str, Any] | None = None, max_attempts: int = 3,
        use_cache: bool = True, image_dimensions=None, extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        culture_metadata = self._culture_metadata(project_id)
        operation_metadata = {**operation.metadata(), **culture_metadata, **(extra_metadata or {})}
        if use_cache:
            lookup_started = time.perf_counter()
            cached = self.repository.get_cached_model_run(cache_key)
            with observer.span(
                "cache.lookup", project_id=project_id,
                metadata={
                    "operation": operation.name, "hit": bool(cached),
                    "latency_ms": int((time.perf_counter() - lookup_started) * 1000),
                },
            ) as cache_observation:
                observer.safe_update(cache_observation, output={"hit": bool(cached)})
            if cached:
                self.repository.record_model_cache_hit(cache_key)
                observer.record_model_call(cache_hit=True)
                return output_builder(None, cached["output"])[1]

        existing = self.repository.get_model_run(cache_key)
        run_id = existing["id"] if existing else str(uuid.uuid4())
        attempt_offset = int((existing or {}).get("attempt_count") or 0)
        started = time.perf_counter()
        input_preview = preview_value(input_data)
        aggregate_usage: dict[str, int] = {}
        aggregate_cost: dict[str, float] = {}
        last_error: Exception | None = None
        final_telemetry = ProviderTelemetry()
        final_trace_id: str | None = None
        final_observation_id: str | None = None

        with observer.span(
            f"ai.{operation.name}", project_id=project_id,
            metadata={**operation_metadata, "cache_key": cache_key, "cache_hit": False},
            input_data={"operation": operation.name, "input": input_preview},
        ) as logical_observation:
            for local_attempt in range(1, max_attempts + 1):
                attempt = attempt_offset + local_attempt
                attempt_started_at = utc_now()
                attempt_started = time.perf_counter()
                provider_result: ProviderResult[Any] | None = None
                attempt_error: Exception | None = None
                telemetry = ProviderTelemetry()
                estimate = gemini_pricing.estimate(model, telemetry.usage)
                trace_id: str | None = None
                observation_id: str | None = None
                with observer.generation(
                    operation.name, project_id=project_id, model=model,
                    input_data={
                        "operation": operation.name,
                        "schema": operation.schema_name,
                        "prompt": input_preview,
                    },
                    metadata={
                        **operation_metadata, "provider": self.provider.name, "attempt": attempt,
                        "retry": local_attempt > 1, "cache_key": cache_key, "prompt_version": PROMPT_VERSION,
                        "capture_mode": settings.langfuse_content_mode,
                    },
                    model_parameters=model_parameters,
                ) as generation:
                    try:
                        provider_result = invoke()
                        telemetry = provider_result.telemetry
                        dimensions = image_dimensions(provider_result) if image_dimensions else None
                        estimate = gemini_pricing.estimate(
                            model, telemetry.usage, on_date=date.today(), image_dimensions=dimensions,
                        )
                        persisted_output, return_value = output_builder(provider_result, None)
                        observer.safe_update(
                            generation,
                            output=preview_value(persisted_output),
                            usage_details=telemetry.usage.langfuse_details(),
                            cost_details=estimate.details,
                            metadata={
                                **operation_metadata, "attempt": attempt,
                                "result": "success",
                                "provider_latency_ms": telemetry.provider_latency_ms,
                                "parse_latency_ms": telemetry.parse_latency_ms,
                                "io_latency_ms": telemetry.io_latency_ms,
                                "latency_ms": int((time.perf_counter() - attempt_started) * 1000),
                                "model_version": telemetry.model_version,
                                "response_id": telemetry.response_id,
                                "finish_reason": telemetry.finish_reason,
                                "cost_basis": estimate.cost_basis,
                                "cost_calculation_basis": estimate.calculation_basis,
                                "pricing_catalog_version": estimate.catalog_version,
                                "pricing_tier": settings.gemini_pricing_tier,
                                "actual_invoice_cost": "unavailable",
                                "free_allowance_applied": estimate.free_allowance_applied,
                                "time_to_first_token": "unavailable_non_streaming",
                            },
                        )
                    except Exception as error:
                        attempt_error = error
                        if isinstance(error, ProviderAttemptError):
                            telemetry = error.telemetry
                        elif provider_result is not None:
                            telemetry = provider_result.telemetry
                        else:
                            telemetry = ProviderTelemetry()
                        estimate = gemini_pricing.estimate(model, telemetry.usage, on_date=date.today())
                        observer.safe_update(
                            generation, level="ERROR", status_message=str(error),
                            usage_details=telemetry.usage.langfuse_details(), cost_details=estimate.details,
                            metadata={
                                **operation_metadata, "attempt": attempt,
                                "result": "failed",
                                "provider_latency_ms": telemetry.provider_latency_ms,
                                "parse_latency_ms": telemetry.parse_latency_ms,
                                "io_latency_ms": telemetry.io_latency_ms,
                                "latency_ms": int((time.perf_counter() - attempt_started) * 1000),
                                "error_type": type(error).__name__, "cost_basis": estimate.cost_basis,
                                "pricing_catalog_version": estimate.catalog_version,
                                "pricing_tier": settings.gemini_pricing_tier,
                            },
                        )
                    trace_id, observation_id = observer.identifiers(generation)

                usage_details = telemetry.usage.langfuse_details()
                for key, value in usage_details.items():
                    aggregate_usage[key] = aggregate_usage.get(key, 0) + value
                for key, value in estimate.details.items():
                    aggregate_cost[key] = aggregate_cost.get(key, 0.0) + value
                observer.record_model_call(
                    usage=usage_details, cost=estimate.total, failed=bool(attempt_error), retry=local_attempt > 1,
                )
                self.repository.save_model_attempt(
                    model_run_id=run_id, project_id=project_id, invocation_id=observer.invocation_id,
                    operation=operation.name, attempt=attempt, provider=self.provider.name, model=model,
                    status="failed" if attempt_error else "success", started_at=attempt_started_at,
                    completed_at=utc_now(), latency_ms=int((time.perf_counter() - attempt_started) * 1000),
                    provider_latency_ms=telemetry.provider_latency_ms, parse_latency_ms=telemetry.parse_latency_ms,
                    io_latency_ms=telemetry.io_latency_ms,
                    usage=telemetry.usage.as_dict(), cost=estimate.details, estimated_cost_usd=estimate.total,
                    model_version=telemetry.model_version, response_id=telemetry.response_id,
                    finish_reason=telemetry.finish_reason,
                    error_type=type(attempt_error).__name__ if attempt_error else None,
                    error=str(attempt_error) if attempt_error else None,
                    langfuse_trace_id=trace_id, langfuse_observation_id=observation_id,
                )
                final_telemetry = telemetry
                final_trace_id, final_observation_id = trace_id, observation_id
                if not attempt_error:
                    output_preview = preview_value(persisted_output)
                    total_cost = round(sum(aggregate_cost.values()), 9)
                    self.repository.save_model_run(
                        id=run_id, project_id=project_id, operation=operation.name, provider=self.provider.name,
                        model=model, prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
                        output=persisted_output, status="success", **culture_metadata,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        input_tokens=aggregate_usage.get("input"), output_tokens=aggregate_usage.get("output"),
                        total_tokens=aggregate_usage.get("total"), operation_metadata=operation_metadata,
                        input=input_data, input_preview=input_preview, output_preview=output_preview,
                        usage=aggregate_usage, cost=aggregate_cost, estimated_cost_usd=total_cost,
                        pricing_catalog_version=gemini_pricing.version, cost_basis=estimate.cost_basis,
                        cost_calculation_basis=estimate.calculation_basis,
                        model_version=telemetry.model_version, response_id=telemetry.response_id,
                        finish_reason=telemetry.finish_reason, attempt_count=attempt,
                        cache_hit_count=int((existing or {}).get("cache_hit_count") or 0),
                        langfuse_trace_id=trace_id, langfuse_observation_id=observation_id,
                    )
                    observer.safe_update(
                        logical_observation, output={"status": "success", "attempts": local_attempt},
                        metadata={
                            **operation_metadata, "attempts": local_attempt, "usage": aggregate_usage,
                            "estimated_cost_usd": total_cost,
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                        },
                    )
                    return return_value

                last_error = attempt_error
                if local_attempt < max_attempts:
                    delay = 1.5 * local_attempt
                    with observer.span(
                        "retry.backoff", project_id=project_id,
                        metadata={"operation": operation.name, "attempt": attempt, "delay_seconds": delay},
                    ):
                        time.sleep(delay)

        total_cost = round(sum(aggregate_cost.values()), 9)
        self.repository.save_model_run(
            id=run_id, project_id=project_id, operation=operation.name, provider=self.provider.name,
            model=model, prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
            status="failed", **culture_metadata,
            latency_ms=int((time.perf_counter() - started) * 1000), input_tokens=aggregate_usage.get("input"),
            output_tokens=aggregate_usage.get("output"), total_tokens=aggregate_usage.get("total"),
            operation_metadata=operation_metadata, input=input_data, input_preview=input_preview,
            output_preview=None, usage=aggregate_usage, cost=aggregate_cost, estimated_cost_usd=total_cost,
            pricing_catalog_version=gemini_pricing.version, cost_basis="public_list_price_estimate",
            cost_calculation_basis="provider_usage", model_version=final_telemetry.model_version,
            response_id=final_telemetry.response_id, finish_reason=final_telemetry.finish_reason,
            attempt_count=attempt_offset + max_attempts,
            cache_hit_count=int((existing or {}).get("cache_hit_count") or 0),
            langfuse_trace_id=final_trace_id, langfuse_observation_id=final_observation_id,
            error=str(last_error),
        )
        raise RuntimeError(
            f"Model operation {operation.name} failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def cached_structured(self, project_id: str, operation: str, prompt: str, schema: type[T]) -> T:
        descriptor = ModelOperation.from_legacy(operation, schema_name=schema.__name__)
        prompt_hash = content_hash(f"{PROMPT_VERSION}:{prompt}")
        input_hash = content_hash(prompt)
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        return self._execute(
            project_id=project_id, operation=descriptor, model=self.provider.text_model,
            prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
            input_data={"prompt": prompt, "schema": schema.__name__},
            invoke=lambda: self.provider.structured(prompt, schema),
            output_builder=lambda result, cached: (
                (result.value.model_dump(mode="json") if result else cached),
                schema.model_validate(result.value if result else cached),
            ),
            model_parameters={"response_mime_type": "application/json", "schema": schema.__name__},
        )

    def cached_research(self, project_id: str, operation: str, prompt: str) -> str:
        descriptor = ModelOperation.from_legacy(operation)
        prompt_hash = content_hash(f"{PROMPT_VERSION}:{prompt}")
        input_hash = content_hash(prompt)
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        def research_output(result: ProviderResult[Any] | None, cached: Any):
            if result is None:
                return cached, cached["text"]
            sources = [
                line.removeprefix("- ").strip() for line in result.value.split("\n")
                if line.strip().startswith("- ") and "http" in line
            ]
            return {"text": result.value, "sources": sources}, result.value

        return self._execute(
            project_id=project_id, operation=descriptor, model=self.provider.text_model,
            prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
            input_data={"prompt": prompt, "tool": "google_search"},
            invoke=lambda: self.provider.grounded_research(prompt),
            output_builder=research_output,
            model_parameters={"tools": ["google_search"]},
        )

    def cached_visual_verification(
        self, project_id: str, asset_id: str, dependency_hash: str, prompt: str,
        image: Path, references: list[ImageReference] | None = None,
    ) -> VisualVerification:
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        prompt_hash = content_hash(f"{PROMPT_VERSION}:verify-visual:{prompt}")
        input_hash = content_hash({
            "asset_id": asset_id, "dependency_hash": dependency_hash, "image_hash": image_hash,
            "references": [
                {"label": label, "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
                for label, path in references or []
            ],
        })
        operation = f"verify-visual-{asset_id}"
        descriptor = ModelOperation(
            name="verify-visual", modality="vision", cache_scope=operation,
            target_type="asset", target_id=asset_id, schema_name="VisualVerification",
        )
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        asset_metadata = self._asset_metadata(project_id, asset_id)
        descriptor = ModelOperation(
            name=descriptor.name, modality=descriptor.modality, cache_scope=descriptor.cache_scope,
            target_type=descriptor.target_type, target_id=descriptor.target_id,
            scene_number=asset_metadata.get("scene_number"), schema_name=descriptor.schema_name,
        )
        references_metadata = self._reference_metadata(references)
        return self._execute(
            project_id=project_id, operation=descriptor, model=self.provider.text_model,
            prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
            input_data={
                "asset_id": asset_id, "dependency_hash": dependency_hash, "image_hash": image_hash,
                "specification": prompt, "references": references_metadata,
            },
            invoke=lambda: self.provider.verify_image(asset_id, dependency_hash, prompt, image, references),
            output_builder=lambda result, cached: (
                result.value.model_dump(mode="json") if result else cached,
                VisualVerification.model_validate(result.value if result else cached),
            ),
            model_parameters={"response_mime_type": "application/json", "temperature": 0},
            extra_metadata=asset_metadata,
        )

    def generate_image(
        self, project_id: str, asset_id: str, prompt: str, output: Path,
        references: list[ImageReference] | None = None,
    ) -> None:
        operation = f"generate-image-{asset_id}"
        asset_metadata = self._asset_metadata(project_id, asset_id)
        descriptor = ModelOperation(
            name="generate-image", modality="image", cache_scope=operation,
            target_type="asset", target_id=asset_id, scene_number=asset_metadata.get("scene_number"),
        )
        prompt_hash = content_hash(f"{PROMPT_VERSION}:image:{prompt}")
        input_hash = content_hash({
            "asset_id": asset_id,
            "references": [
                {"label": label, "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
                for label, path in references or []
            ],
        })
        cache_key = content_hash(
            f"{project_id}:{self.provider.name}:{self.provider.image_model}:{operation}:{prompt_hash}:{input_hash}"
        )
        references_metadata = self._reference_metadata(references)

        def build_output(result: ProviderResult[Any] | None, cached: Any):
            if result is None:
                return cached, None
            validation_started = time.perf_counter()
            with Image.open(output) as generated:
                metadata = {
                    **result.value, "image_hash": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "width": generated.width, "height": generated.height, "format": generated.format,
                }
            validation_ms = int((time.perf_counter() - validation_started) * 1000)
            result.telemetry.io_latency_ms = (result.telemetry.io_latency_ms or 0) + validation_ms
            return metadata, None

        def generated_dimensions(_: ProviderResult[Any]) -> tuple[int, int]:
            with Image.open(output) as generated:
                return generated.width, generated.height

        self._execute(
            project_id=project_id, operation=descriptor, model=self.provider.image_model,
            prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key,
            input_data={"prompt": prompt, "references": references_metadata},
            invoke=lambda: self.provider.generate_image(prompt, output, references), output_builder=build_output,
            model_parameters={"reference_count": len(references or [])}, max_attempts=1, use_cache=False,
            image_dimensions=generated_dimensions, extra_metadata=asset_metadata,
        )
