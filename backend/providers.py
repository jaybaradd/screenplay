from __future__ import annotations

import hashlib
import re
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from PIL import Image, ImageDraw
from pydantic import BaseModel

from backend.config import settings
from backend.observability import observer
from backend.prompts import PROMPT_VERSION
from backend.schemas import (
    AdaptedBlock, AdaptedScene, AdaptedScreenplay, BlockType, CharacterRecord,
    ContentBlock, CulturalBrief, CulturalClaim, DialectGuide, LayerDecision, LayerPlan,
    ProductionElement, SceneRecord, SourceScreenplay, StoryContract,
    VisualVerification,
)
from backend.storage import Repository, content_hash


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
    def structured(self, prompt: str, schema: type[T]) -> T: ...

    @abstractmethod
    def grounded_research(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_image(self, prompt: str, output: Path, references: list[ImageReference] | None = None) -> None: ...

    @abstractmethod
    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> VisualVerification: ...


class GeminiProvider(Provider):
    name = "google"

    def __init__(self) -> None:
        from google import genai

        self.text_model = settings.text_model
        self.image_model = settings.image_model
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def structured(self, prompt: str, schema: type[T]) -> T:
        from google.genai import types

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.text_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=gemini_json_schema(schema),
                    ),
                )
                return parse_structured_response(response, schema)
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Gemini structured output failed after 3 attempts: {last_error}")

    def grounded_research(self, prompt: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.text_model,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        sources: list[str] = []
        for candidate in response.candidates or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if uri:
                    sources.append(f"- {title or 'Source'}: {uri}")
        return f"{response.text or ''}\n\nSOURCES\n" + "\n".join(dict.fromkeys(sources))

    def generate_image(self, prompt: str, output: Path, references: list[ImageReference] | None = None) -> None:
        from google.genai import types

        contents: list[Any] = [prompt]
        for label, path in references or []:
            contents.append(f"REFERENCE IMAGE — {label}")
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
        response = self.client.models.generate_content(model=self.image_model, contents=contents)
        for part in response.parts or []:
            if getattr(part, "inline_data", None):
                output.write_bytes(part.inline_data.data)
                return
        raise RuntimeError("Gemini returned no image data")

    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> VisualVerification:
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
        last_error: Exception | None = None
        result: VisualVerification | None = None
        for attempt in range(3):
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
                result = parse_structured_response(response, VisualVerification)
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if result is None:
            raise RuntimeError(f"Gemini visual verification failed after 3 attempts: {last_error}") from last_error
        if result.asset_id != asset_id or result.dependency_hash != dependency_hash:
            raise ValueError("Visual verifier returned mismatched asset identity")
        return result


class MockProvider(Provider):
    name = "mock"
    text_model = "deterministic-mock-v1"
    image_model = "deterministic-mock-image-v1"

    def __init__(self) -> None:
        self.context: dict[str, Any] = {}

    def set_context(self, **values: Any) -> None:
        self.context.update(values)

    def structured(self, prompt: str, schema: type[T]) -> T:
        if schema is CulturalBrief:
            return self._cultural_brief()  # type: ignore[return-value]
        if schema is LayerPlan:
            match = re.search(r"layer=(\w+)", prompt)
            return self._layer_plan(match.group(1) if match else "verbal")  # type: ignore[return-value]
        raise NotImplementedError(f"Mock structured output not implemented for {schema.__name__}")

    def grounded_research(self, prompt: str) -> str:
        return (
            "MOCK MODE: No web research was performed. The brief will contain only cautious workflow constraints, "
            "and must not be presented as verified cultural evidence. Configure GEMINI_API_KEY and AI_MODE=live for grounding."
        )

    def generate_image(self, prompt: str, output: Path, references: list[ImageReference] | None = None) -> None:
        digest = content_hash(prompt)
        colour = tuple(int(digest[index:index + 2], 16) for index in (0, 2, 4))
        image = Image.new("RGB", (1024, 1024), colour)
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 60, 964, 964), outline="white", width=6)
        draw.text((90, 100), "MVP MOCK VISUAL", fill="white")
        draw.text((90, 145), content_hash(prompt)[:16], fill="white")
        draw.multiline_text((90, 210), prompt[:700], fill="white", spacing=8)
        image.save(output, format="PNG")

    def verify_image(
        self, asset_id: str, dependency_hash: str, prompt: str, image: Path,
        references: list[ImageReference] | None = None,
    ) -> VisualVerification:
        return VisualVerification(
            asset_id=asset_id, dependency_hash=dependency_hash, passed=True, issues=[],
            summary="Mock verifier confirms only workflow plumbing, not visual or cultural accuracy.",
        )

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
        return CulturalBrief(
            culture="Maidani Mewari", locality=self.context.get("locality", "User-selected locality"),
            setting=self.context.get("setting", "rural"), period=self.context.get("period", "Contemporary"),
            claims=[CulturalClaim(
                id="mock-claim", claim="No cultural factual claim is asserted in mock mode.",
                scope="Workflow demonstration only", time_period="Not applicable", source_url="mock://no-live-grounding",
                confidence="low", layers=["cultural_precision"],
                uncertainty="Configure live Gemini grounding before presenting cultural output.",
                prohibited_extrapolations=["Do not treat mock content as cultural evidence."],
            )],
            dialect_guide=DialectGuide(
                target_variety="Maidani Mewari", writing_script="Devanagari", features=[],
                register_rules=[], code_switching_rules=[],
                negative_constraints=["Mock mode cannot supply culturally verified dialogue."],
            ),
            negative_constraints=[
                "Do not default to palaces, camels, sand dunes, weddings or tourist folk imagery.",
                "Do not invent Mewari dialogue or infer caste, religion or class.",
            ],
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
                risks=["Mock output is not culturally validated"], do_not_change=["Core story logic"],
                negative_constraints=["Do not invent culture-specific detail"],
            )],
        )


class AIService:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.provider: Provider = GeminiProvider() if settings.ai_mode == "live" and settings.gemini_api_key else MockProvider()

    def cached_structured(self, project_id: str, operation: str, prompt: str, schema: type[T]) -> T:
        prompt_hash = content_hash(f"{PROMPT_VERSION}:{prompt}")
        input_hash = content_hash(prompt)
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        cached = self.repository.get_cached_model_run(cache_key)
        if cached:
            return schema.model_validate(cached["output"])
        started = time.perf_counter()
        try:
            with observer.generation(operation, project_id=project_id, model=self.provider.text_model, input_data=prompt) as observation:
                result = self.provider.structured(prompt, schema)
                output = result.model_dump(mode="json")
                observation.update(output=output if settings.langfuse_capture_content else {"output_hash": content_hash(output)})
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, output=output, status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return result
        except Exception as error:
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000), error=str(error),
            )
            raise

    def cached_research(self, project_id: str, operation: str, prompt: str) -> str:
        prompt_hash = content_hash(f"{PROMPT_VERSION}:{prompt}")
        input_hash = content_hash(prompt)
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        cached = self.repository.get_cached_model_run(cache_key)
        if cached:
            return cached["output"]["text"]
        started = time.perf_counter()
        try:
            with observer.generation(operation, project_id=project_id, model=self.provider.text_model, input_data=prompt) as observation:
                result = self.provider.grounded_research(prompt)
                observation.update(output=result if settings.langfuse_capture_content else {"output_hash": content_hash(result)})
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, output={"text": result}, status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return result
        except Exception as error:
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000), error=str(error),
            )
            raise

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
        cache_key = content_hash(f"{project_id}:{self.provider.name}:{self.provider.text_model}:{operation}:{prompt_hash}:{input_hash}")
        cached = self.repository.get_cached_model_run(cache_key)
        if cached:
            return VisualVerification.model_validate(cached["output"])
        started = time.perf_counter()
        try:
            with observer.generation(operation, project_id=project_id, model=self.provider.text_model, input_data={
                "asset_id": asset_id, "dependency_hash": dependency_hash, "image_hash": image_hash,
            }) as observation:
                result = self.provider.verify_image(asset_id, dependency_hash, prompt, image, references)
                output = result.model_dump(mode="json")
                observation.update(output=output)
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, output=output, status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return result
        except Exception as error:
            self.repository.save_model_run(
                project_id=project_id, operation=operation, provider=self.provider.name, model=self.provider.text_model,
                prompt_hash=prompt_hash, input_hash=input_hash, cache_key=cache_key, status="failed",
                latency_ms=int((time.perf_counter() - started) * 1000), error=str(error),
            )
            raise
