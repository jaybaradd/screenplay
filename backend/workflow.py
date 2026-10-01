from __future__ import annotations

import operator
import os
import sqlite3
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from backend.config import settings
from backend.adaptation_validation import language_audit_issues, validate_output_script
from backend.cultures.context import CultureContextCompiler
from backend.cultures.models import CultureProfileSnapshot, TargetSelection
from backend.continuity import check_continuity
from backend.state_ledger import build_state_ledger
from backend.observability import observer
from backend.prompts import (
    EXTRACTION_PROMPT, adaptation_prompt, adaptation_repair_prompt, cultural_normalize_prompt,
    cultural_research_prompt, language_audit_prompt, layer_prompt, synthesis_prompt,
    visual_manifest_prompt, VISUAL_STYLE_LOCK,
)
from backend.providers import AIService, MockProvider, stable_id
from backend.schemas import (
    AdaptationPlan, AdaptedScene, AdaptedScreenplay, AppearanceSpec, CulturalBrief, CulturalClaim, CulturalConstraint,
    LanguageAudit, LanguageFeature, LayerPlan, SceneRecord, SceneVisualSpec, SetVisualSpec, SourceScreenplay, VisualManifest,
)
from backend.storage import Repository, content_hash


class GraphState(TypedDict, total=False):
    project_id: str
    extraction_revision: str
    issues_revision: str
    cultural_revision: str
    layer_revision_ids: Annotated[list[str], operator.add]
    plan_revision: str
    adapted_revision: str
    visual_revision: str
    stage: str
    error: str


LAYER_NAMES = ["verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision"]


class Workflow:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.ai = AIService(repository)
        settings.ensure_directories()
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        self._checkpoint_connection = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("validate", self._traced("validate", self.validate))
        builder.add_node("extract", self._traced("extract", self.extract))
        builder.add_node("continuity", self._traced("continuity", self.continuity))
        builder.add_node("review_extraction", self._traced("review_extraction", self.review_extraction))
        builder.add_node("culture", self._traced("culture", self.culture))
        for layer in LAYER_NAMES:
            builder.add_node(f"plan_{layer}", self._traced(f"plan_{layer}", self._layer_node(layer)))
        builder.add_node("synthesize_plan", self._traced("synthesize_plan", self.synthesize_plan))
        builder.add_node("review_plan", self._traced("review_plan", self.review_plan))
        builder.add_node("adapt", self._traced("adapt", self.adapt))
        builder.add_node("verify", self._traced("verify", self.verify))
        builder.add_node("review_screenplay", self._traced("review_screenplay", self.review_screenplay))
        builder.add_node("visual_manifest", self._traced("visual_manifest", self.visual_manifest))
        builder.add_node("review_visuals", self._traced("review_visuals", self.review_visuals))
        builder.add_node("prepare_character_assets", self._traced("prepare_character_assets", self.prepare_character_assets))
        builder.add_node("review_character_images", self._traced("review_character_images", self.review_character_images))
        builder.add_node("prepare_scene_assets", self._traced("prepare_scene_assets", self.prepare_scene_assets))

        builder.add_edge(START, "validate")
        builder.add_edge("validate", "extract")
        builder.add_edge("extract", "continuity")
        builder.add_edge("continuity", "review_extraction")
        builder.add_edge("review_extraction", "culture")
        for layer in LAYER_NAMES:
            builder.add_edge("culture", f"plan_{layer}")
            builder.add_edge(f"plan_{layer}", "synthesize_plan")
        builder.add_edge("synthesize_plan", "review_plan")
        builder.add_edge("review_plan", "adapt")
        builder.add_edge("adapt", "verify")
        builder.add_edge("verify", "review_screenplay")
        builder.add_edge("review_screenplay", "visual_manifest")
        builder.add_edge("visual_manifest", "review_visuals")
        builder.add_edge("review_visuals", "prepare_character_assets")
        builder.add_edge("prepare_character_assets", "review_character_images")
        builder.add_edge("review_character_images", "prepare_scene_assets")
        builder.add_edge("prepare_scene_assets", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _traced(name: str, node):
        def wrapped(state: GraphState):
            if name.startswith("plan_"):
                span_name = "graph.plan_layer"
            elif name in {"prepare_character_assets", "prepare_scene_assets"}:
                span_name = "graph.prepare_assets"
            else:
                span_name = f"graph.{name}"
            metadata = {"node": name}
            if name.startswith("plan_"):
                metadata["layer"] = name.removeprefix("plan_")
            if name.startswith("prepare_"):
                metadata["asset_kind"] = name.removeprefix("prepare_").removesuffix("_assets")
            with observer.span(span_name, project_id=state["project_id"], metadata=metadata):
                return node(state)
        return wrapped

    def config(self, project_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": project_id}}

    def _culture_compiler(
        self, project_id: str, brief_revision: dict[str, Any] | None = None,
    ) -> CultureContextCompiler:
        project = self.repository.get_project(project_id)
        snapshot_revision = self.repository.latest_revision(project_id, "culture_profile_snapshot")
        if not snapshot_revision:
            raise ValueError("Project has no immutable culture profile snapshot")
        snapshot = CultureProfileSnapshot.model_validate(snapshot_revision["payload"])
        if snapshot.profile_hash != project["profile_hash"]:
            raise ValueError("Project culture profile hash does not match its immutable snapshot")
        selection = TargetSelection(
            culture_id=project["culture_id"], locality=project["locality"], setting=project["setting"],
            period=project["period"], output_script=project["output_script"],
        )
        return CultureContextCompiler(
            snapshot, selection, brief_revision["payload"] if brief_revision else None,
            brief_revision_id=brief_revision["id"] if brief_revision else None,
            brief_hash=brief_revision["sha256"] if brief_revision else None,
        )

    @staticmethod
    def _bind_brief_identity(brief: CulturalBrief, context) -> CulturalBrief:
        brief.culture_id = context.culture_id
        brief.profile_version = context.profile_version
        brief.profile_hash = context.profile_hash
        brief.culture = context.display_name
        brief.locality = context.locality
        brief.setting = context.setting
        brief.period = context.period
        brief.output_script = context.output_script
        claims = {item.id: item for item in brief.claims}
        for item in context.reviewed_evidence.claims:
            reviewed = item.model_dump(mode="json")
            reviewed["layers"] = reviewed.pop("permitted_layers")
            claim = CulturalClaim.model_validate({**reviewed, "origin": "reviewed_profile_evidence"})
            claims[claim.id] = claim
        brief.claims = list(claims.values())
        if brief.language_guide:
            features = {item.id: item for item in brief.language_guide.features}
            for item in context.reviewed_evidence.language_features:
                feature = LanguageFeature.model_validate(item.model_dump(mode="json"))
                features[feature.id] = feature
            brief.language_guide.features = list(features.values())
        constraints = {item.id: item for item in brief.constraints}
        for item in context.constraints:
            if item.get("origin") != "profile_policy":
                continue
            constraints[item["id"]] = CulturalConstraint(
                id=item["id"], text=item["text"], layers=item["layers"],
                kind=item["kind"], origin="profile_policy", source_claim_ids=[],
            )
        brief.constraints = list(constraints.values())
        return brief

    def start(self, project_id: str) -> dict[str, Any]:
        with observer.span("workflow.start", project_id=project_id):
            return self.graph.invoke(
                {"project_id": project_id, "layer_revision_ids": [], "stage": "created"},
                self.config(project_id),
            )

    def resume(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with observer.span("workflow.resume", project_id=project_id, metadata={"action": payload.get("action")}):
            return self.graph.invoke(Command(resume=payload), self.config(project_id))

    def retry_failed(self, project_id: str) -> dict[str, Any]:
        with observer.span("workflow.retry", project_id=project_id):
            return self.graph.invoke(None, self.config(project_id))

    def validate(self, state: GraphState) -> GraphState:
        project = self.repository.get_project(state["project_id"])
        text = project["source_text"].strip()
        if len(text) < 100 or len(text) > 40_000:
            raise ValueError("Source screenplay must contain 100-40,000 characters")
        self.repository.set_stage(state["project_id"], "extracting")
        return {"stage": "extracting"}

    def extract(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        existing = self.repository.latest_revision(project_id, "extraction")
        if existing:
            return {"extraction_revision": existing["id"]}
        project = self.repository.get_project(project_id)
        if isinstance(self.ai.provider, MockProvider):
            document = self.ai.provider.extract(project_id, project["title"], project["source_text"])
        else:
            prompt = f"{EXTRACTION_PROMPT}\nPROJECT ID: {project_id}\nSOURCE:\n{project['source_text']}"
            document = self.ai.cached_structured(project_id, "extract-screenplay", prompt, SourceScreenplay)
            document = self._normalize_ids(project_id, document)
        revision = self.repository.create_revision(project_id, "extraction", document)
        return {"extraction_revision": revision["id"]}

    def continuity(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        source = SourceScreenplay.model_validate(self.repository.get_revision(state["extraction_revision"])["payload"])
        issues = check_continuity(source)
        observer.score("extraction_completeness", 1.0 if source.scenes else 0.0, project_id=project_id)
        observer.score("continuity_violations", float(len(issues)), project_id=project_id)
        observer.score(
            "continuity_blocking_count", float(sum(item.severity == "blocking" for item in issues)),
            project_id=project_id,
        )
        revision = self.repository.create_revision(project_id, "continuity_issues", [item.model_dump(mode="json") for item in issues])
        payload = {
            "gate": "extraction", "message": "Review extraction, canonical records and story contract.",
            "extraction_revision": state["extraction_revision"], "issues_revision": revision["id"],
            "blocking_issues": sum(item.severity == "blocking" for item in issues),
        }
        self.repository.set_stage(project_id, "extraction_review", interrupt=payload)
        return {"issues_revision": revision["id"], "stage": "extraction_review"}

    def review_extraction(self, state: GraphState) -> GraphState:
        decision = interrupt({"gate": "extraction", "project_id": state["project_id"]})
        if decision.get("action") != "approve":
            raise ValueError("Extraction must be approved to continue")
        self.repository.set_stage(state["project_id"], "researching_culture")
        return {"extraction_revision": decision.get("revision_id") or state["extraction_revision"], "stage": "researching_culture"}

    def culture(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        existing = self.repository.latest_revision(project_id, "cultural_brief")
        if existing:
            return {"cultural_revision": existing["id"], "layer_revision_ids": []}
        compiler = self._culture_compiler(project_id)
        context = compiler.for_research()
        if isinstance(self.ai.provider, MockProvider):
            self.ai.provider.set_context(culture_context=context.model_dump(mode="json"))
        research_prompt = cultural_research_prompt(context)
        research = self.ai.cached_research(project_id, "cultural-grounding", research_prompt)
        normalize_prompt = cultural_normalize_prompt(context, research)
        brief = self.ai.cached_structured(project_id, "normalize-cultural-brief", normalize_prompt, CulturalBrief)
        brief = self._bind_brief_identity(brief, context)
        revision = self.repository.create_revision(project_id, "cultural_brief", brief)
        return {"cultural_revision": revision["id"], "layer_revision_ids": []}

    def _layer_node(self, layer: str):
        def node(state: GraphState) -> GraphState:
            project_id = state["project_id"]
            existing = self.repository.latest_revision(project_id, f"plan_{layer}")
            if existing:
                return {"layer_revision_ids": [existing["id"]]}
            source = self.repository.get_revision(state["extraction_revision"])["payload"]
            brief_revision = self.repository.get_revision(state["cultural_revision"])
            context = self._culture_compiler(project_id, brief_revision).for_layer(layer)
            result = self.ai.cached_structured(
                project_id, f"plan-{layer}", layer_prompt(layer, source, context), LayerPlan,
            )
            revision = self.repository.create_revision(project_id, f"plan_{layer}", result)
            return {"layer_revision_ids": [revision["id"]]}
        return node

    def synthesize_plan(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        existing = self.repository.latest_revision(project_id, "adaptation_plan")
        if existing:
            plan_revision = existing
        else:
            layer_revisions = [self.repository.get_revision(revision_id) for revision_id in dict.fromkeys(state["layer_revision_ids"])]
            layers = [LayerPlan.model_validate(revision["payload"]) for revision in layer_revisions]
            layers.sort(key=lambda layer: LAYER_NAMES.index(layer.layer))
            if isinstance(self.ai.provider, MockProvider):
                plan = AdaptationPlan(
                    layers=layers,
                    synthesis=[
                        "Apply only source-backed changes approved in the six layer plans.",
                        "StoryContract invariants override decorative cultural additions.",
                        "Uncertain language remains restrained and visibly marked.",
                    ],
                    deeper_change_requests=[],
                    story_preservation_notes=["Preserve scene order, causal events, relationships and emotional arc."],
                )
            else:
                source = self.repository.get_revision(state["extraction_revision"])["payload"]
                plan = self.ai.cached_structured(
                    project_id, "synthesize-adaptation-plan",
                    synthesis_prompt(
                        [layer.model_dump(mode="json") for layer in layers], source["story_contract"],
                    ),
                    AdaptationPlan,
                )
                if {layer.layer for layer in plan.layers} != set(LAYER_NAMES):
                    raise ValueError("Plan synthesis did not preserve all six required layers")
            plan_revision = self.repository.create_revision(project_id, "adaptation_plan", plan)
        source = SourceScreenplay.model_validate(self.repository.get_revision(state["extraction_revision"])["payload"])
        brief_revision = self.repository.get_revision(state["cultural_revision"])
        brief = CulturalBrief.model_validate(brief_revision["payload"])
        plan = AdaptationPlan.model_validate(plan_revision["payload"])
        context = self._culture_compiler(project_id, brief_revision).for_language_audit()
        plan_issues = self._plan_quality_issues(source, brief, plan, context)
        previous_verification = self.repository.latest_revision(project_id, "plan_verification")
        verification_revision = self.repository.create_revision(
            project_id, "plan_verification", plan_issues,
            previous_verification["id"] if previous_verification else None,
        )
        payload = {
            "gate": "plan", "message": "Review cultural brief, six adaptation layers and canonical records.",
            "plan_revision": plan_revision["id"], "cultural_revision": state["cultural_revision"],
            "verification_revision": verification_revision["id"],
            "blocking_issues": sum(issue["severity"] == "blocking" for issue in plan_issues),
        }
        self.repository.set_stage(project_id, "plan_review", interrupt=payload)
        return {"plan_revision": plan_revision["id"], "stage": "plan_review"}

    def review_plan(self, state: GraphState) -> GraphState:
        decision = interrupt({"gate": "plan", "project_id": state["project_id"]})
        if decision.get("action") != "approve":
            raise ValueError("Adaptation plan must be approved to continue")
        self.repository.set_stage(state["project_id"], "adapting")
        return {"plan_revision": decision.get("revision_id") or state["plan_revision"], "stage": "adapting"}

    def adapt(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        existing = self.repository.latest_revision(project_id, "adapted_screenplay")
        if existing:
            return {"adapted_revision": existing["id"]}
        project = self.repository.get_project(project_id)
        source = SourceScreenplay.model_validate(self.repository.get_revision(state["extraction_revision"])["payload"])
        if isinstance(self.ai.provider, MockProvider):
            adapted = self.ai.provider.adapt(source, project["output_script"])
        else:
            brief_revision = self.repository.get_revision(state["cultural_revision"])
            brief = CulturalBrief.model_validate(brief_revision["payload"])
            compiler = self._culture_compiler(project_id, brief_revision)
            plan = AdaptationPlan.model_validate(self.repository.get_revision(state["plan_revision"])["payload"])
            scene_order = {item.id: index for index, item in enumerate(source.scenes)}
            scenes: list[AdaptedScene] = []
            for scene in source.scenes:
                relevant_layers: list[LayerPlan] = []
                claim_ids: set[str] = set()
                for layer in plan.layers:
                    decisions = [
                        decision for decision in layer.decisions
                        if not decision.affected_scene_ids or scene.id in decision.affected_scene_ids
                    ]
                    if decisions:
                        relevant_layers.append(LayerPlan(layer=layer.layer, objective=layer.objective, decisions=decisions))
                        claim_ids.update(claim_id for decision in decisions for claim_id in decision.cultural_claim_ids)
                brief_payload = brief.model_dump(mode="json")
                brief_payload["claims"] = [claim.model_dump(mode="json") for claim in brief.claims if claim.id in claim_ids]
                plan_payload = plan.model_dump(mode="json")
                plan_payload["layers"] = [layer.model_dump(mode="json") for layer in relevant_layers]
                source_context = {
                    "story_contract": source.story_contract.model_dump(mode="json"),
                    "characters": [
                        character.model_dump(mode="json") for character in source.characters
                        if character.id in scene.character_ids
                    ],
                    "production_elements": [
                        element.model_dump(mode="json") for element in source.production_elements
                        if element.id in scene.production_element_ids
                    ],
                    "incoming_state": [
                        transition.model_dump(mode="json") for transition in source.state_transitions
                        if transition.character_id in scene.character_ids
                        and scene_order.get(transition.scene_id, 10**9) <= scene_order[scene.id]
                    ],
                }
                scene_payload = scene.model_dump(mode="json")
                culture_context = compiler.for_scene_adaptation(scene.id, claim_ids)
                prompt = adaptation_prompt(
                    scene_payload, source_context, brief_payload, plan_payload, culture_context,
                )
                result = self.ai.cached_structured(
                    project_id, f"adapt-scene-{scene.number}-attempt-1", prompt, AdaptedScene,
                )
                errors = self._scene_mapping_errors(scene, result)
                for attempt in range(2, 4):
                    if not errors:
                        break
                    prompt = adaptation_repair_prompt(
                        scene_payload, source_context, brief_payload, plan_payload,
                        culture_context, result.model_dump(mode="json"), errors,
                    )
                    result = self.ai.cached_structured(
                        project_id, f"adapt-scene-{scene.number}-attempt-{attempt}", prompt, AdaptedScene,
                    )
                    errors = self._scene_mapping_errors(scene, result)
                if errors:
                    raise ValueError(
                        f"Scene {scene.number} adaptation failed lossless mapping after 3 attempts: "
                        + "; ".join(errors)
                    )
                scenes.append(result)
            adapted = AdaptedScreenplay(
                title=source.title, output_script=project["output_script"], scenes=scenes,
                preservation_summary="All adapted scenes retain source IDs and are checked against the approved StoryContract.",
            )
        revision = self.repository.create_revision(project_id, "adapted_screenplay", adapted)
        return {"adapted_revision": revision["id"]}

    def verify(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        source = SourceScreenplay.model_validate(self.repository.get_revision(state["extraction_revision"])["payload"])
        adapted = AdaptedScreenplay.model_validate(self.repository.get_revision(state["adapted_revision"])["payload"])
        issues: list[dict[str, Any]] = []
        if [scene.id for scene in source.scenes] != [scene.source_scene_id for scene in adapted.scenes]:
            issues.append({"severity": "blocking", "code": "adapted_scene_mismatch", "message": "Adapted scenes do not map one-to-one in source order."})
        source_blocks = {block.id for scene in source.scenes for block in scene.blocks}
        mapped_sequence = [
            source_id for scene in adapted.scenes for block in scene.blocks for source_id in block.source_block_ids
        ]
        mapped_blocks = set(mapped_sequence)
        missing = sorted(source_blocks - mapped_blocks)
        if missing:
            issues.append({"severity": "blocking", "code": "missing_source_blocks", "message": f"Unmapped source blocks: {missing}"})
        unknown = sorted(mapped_blocks - source_blocks)
        if unknown:
            issues.append({"severity": "blocking", "code": "unknown_source_blocks", "message": f"Unknown source block mappings: {unknown}"})
        duplicates = sorted({block_id for block_id in mapped_sequence if mapped_sequence.count(block_id) > 1})
        if duplicates:
            issues.append({"severity": "blocking", "code": "duplicate_source_blocks", "message": f"Source blocks mapped more than once: {duplicates}"})
        snapshot_revision = self.repository.latest_revision(project_id, "culture_profile_snapshot")
        snapshot = CultureProfileSnapshot.model_validate(snapshot_revision["payload"])
        script = next(item for item in snapshot.profile.supported_scripts if item.id == adapted.output_script)
        issues.extend(validate_output_script(adapted, script))
        if not isinstance(self.ai.provider, MockProvider):
            brief_revision = self.repository.get_revision(state["cultural_revision"])
            brief = CulturalBrief.model_validate(brief_revision["payload"])
            culture_context = self._culture_compiler(project_id, brief_revision).for_language_audit()
            audit = self.ai.cached_structured(
                project_id, "audit-adapted-language",
                language_audit_prompt(
                    adapted.model_dump(mode="json"), brief.model_dump(mode="json"), culture_context,
                ),
                LanguageAudit,
            )
            previous_audit = self.repository.latest_revision(project_id, "language_audit")
            self.repository.create_revision(
                project_id, "language_audit", audit,
                previous_audit["id"] if previous_audit else None,
            )
            issues.extend(language_audit_issues(adapted, audit, culture_context))
            audited_dialogue = sum(len(item.dialogue_block_ids) for item in audit.scenes)
            supported_dialogue = sum(len(item.supported_target_variety_block_ids) for item in audit.scenes)
            observer.score(
                "language_audit_pass_ratio",
                (supported_dialogue / audited_dialogue) if audited_dialogue else 1.0,
                project_id=project_id,
            )
        coverage = (len(mapped_blocks & source_blocks) / len(source_blocks)) if source_blocks else 1.0
        observer.score("story_anchor_coverage", coverage, project_id=project_id)
        observer.score("source_block_mapping_coverage", coverage, project_id=project_id)
        revision = self.repository.create_revision(project_id, "adaptation_verification", issues)
        payload = {
            "gate": "screenplay", "message": "Compare source and adapted screenplay; approve or make surgical corrections.",
            "adapted_revision": state["adapted_revision"], "verification_revision": revision["id"],
            "blocking_issues": sum(issue["severity"] == "blocking" for issue in issues),
        }
        self.repository.set_stage(project_id, "screenplay_review", interrupt=payload)
        return {"stage": "screenplay_review"}

    @staticmethod
    def _scene_mapping_errors(source_scene: SceneRecord, adapted_scene: AdaptedScene) -> list[str]:
        errors: list[str] = []
        if adapted_scene.id != source_scene.id:
            errors.append(f"scene id must be {source_scene.id}, got {adapted_scene.id}")
        if adapted_scene.source_scene_id != source_scene.id:
            errors.append(f"source_scene_id must be {source_scene.id}, got {adapted_scene.source_scene_id}")
        if len(adapted_scene.blocks) != len(source_scene.blocks):
            errors.append(f"expected {len(source_scene.blocks)} blocks, got {len(adapted_scene.blocks)}")
        for index, source_block in enumerate(source_scene.blocks):
            if index >= len(adapted_scene.blocks):
                errors.append(f"missing source block at position {index + 1}: {source_block.id}")
                continue
            block = adapted_scene.blocks[index]
            if block.id != source_block.id:
                errors.append(f"block {index + 1} id must be {source_block.id}, got {block.id}")
            if block.source_block_ids != [source_block.id]:
                errors.append(
                    f"block {index + 1} source_block_ids must be [{source_block.id}], got {block.source_block_ids}"
                )
            if block.type != source_block.type:
                errors.append(f"block {source_block.id} changed type from {source_block.type} to {block.type}")
            if block.speaker_id != source_block.speaker_id:
                errors.append(f"block {source_block.id} changed speaker_id")
        return errors

    @staticmethod
    def _plan_quality_issues(
        source: SourceScreenplay, brief: CulturalBrief, plan: AdaptationPlan, context,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if (
            brief.culture_id != context.culture_id
            or brief.profile_hash != context.profile_hash
            or brief.profile_version != context.profile_version
        ):
            issues.append({
                "severity": "blocking", "code": "culture_profile_mismatch",
                "message": "The cultural brief does not match the project's immutable culture profile.",
            })
        guide = brief.language_guide
        if not guide or guide.target_variety.strip().casefold() != context.target_variety.strip().casefold():
            issues.append({
                "severity": "blocking", "code": "missing_target_language_guide",
                "message": f"The cultural brief does not contain a {context.target_variety} language guide.",
            })
            return issues
        if guide.writing_script.strip().casefold() != context.output_script.strip().casefold():
            issues.append({
                "severity": "blocking", "code": "wrong_language_script",
                "message": f"Language guide script must be {context.output_script}, got {guide.writing_script}.",
            })
        usable_features = [feature for feature in guide.features if feature.confidence in {"high", "medium"}]
        minimum_features = context.language_policy.minimum_supported_features
        if len(usable_features) < minimum_features:
            issues.append({
                "severity": "blocking", "code": "insufficient_language_evidence",
                "message": (
                    f"Fewer than {minimum_features} medium/high-confidence, source-backed "
                    f"{context.target_variety} language features are available; "
                    "a native-feeling adaptation cannot be grounded safely."
                ),
            })
        available_categories = {feature.category for feature in usable_features}
        missing_categories = sorted(
            set(context.language_policy.required_feature_categories) - available_categories
        )
        if missing_categories:
            issues.append({
                "severity": "blocking", "code": "missing_language_feature_categories",
                "message": (
                    "The approved language guide is missing required evidence categories: "
                    f"{missing_categories}."
                ),
            })
        verbal = next((layer for layer in plan.layers if layer.layer == "verbal"), None)
        if not verbal or not verbal.decisions:
            issues.append({
                "severity": "blocking", "code": "missing_verbal_plan",
                "message": "The adaptation plan has no usable verbal-layer decisions.",
            })
            return issues
        speaking_scenes = {
            scene.id for scene in source.scenes if any(block.type.value == "dialogue" for block in scene.blocks)
        }
        covered_scenes = {
            scene_id for decision in verbal.decisions for scene_id in decision.affected_scene_ids
        }
        if speaking_scenes - covered_scenes:
            issues.append({
                "severity": "blocking", "code": "verbal_plan_scene_gap",
                "message": f"The verbal plan does not cover speaking scenes: {sorted(speaking_scenes - covered_scenes)}",
            })
        feature_ids = {feature.id for feature in guide.features}
        cited_feature_ids = {
            feature_id for decision in verbal.decisions for feature_id in decision.language_feature_ids
        }
        unknown_feature_ids = sorted(cited_feature_ids - feature_ids)
        if unknown_feature_ids:
            issues.append({
                "severity": "blocking", "code": "verbal_plan_unknown_language_features",
                "message": f"The verbal plan cites unknown LanguageFeature IDs: {unknown_feature_ids}",
            })
        ungrounded_decisions = sorted(
            decision.id for decision in verbal.decisions if not decision.language_feature_ids
        )
        if guide.features and ungrounded_decisions:
            issues.append({
                "severity": "blocking", "code": "verbal_plan_not_grounded",
                "message": (
                    "Verbal-plan decisions must cite approved LanguageFeature IDs. "
                    f"Missing citations: {ungrounded_decisions}"
                ),
            })
        claim_ids = {item.id for item in brief.claims}
        constraint_ids = {item.id for item in brief.constraints}
        if len(claim_ids) != len(brief.claims):
            issues.append({
                "severity": "blocking", "code": "duplicate_cultural_claims",
                "message": "Cultural claim IDs must be unique.",
            })
        reviewed_claim_ids = {item.id for item in context.reviewed_evidence.claims}
        invalid_reviewed_origins = sorted(
            item.id for item in brief.claims
            if item.origin == "reviewed_profile_evidence" and item.id not in reviewed_claim_ids
        )
        if invalid_reviewed_origins:
            issues.append({
                "severity": "blocking", "code": "unapproved_reviewed_evidence",
                "message": f"Claims are marked reviewed but are absent from the profile: {invalid_reviewed_origins}",
            })
        invalid_constraint_sources = sorted(
            item.id for item in brief.constraints
            if item.origin == "grounded_research"
            and (not item.source_claim_ids or not set(item.source_claim_ids).issubset(claim_ids))
        )
        if invalid_constraint_sources:
            issues.append({
                "severity": "blocking", "code": "untraceable_cultural_constraints",
                "message": f"Grounded constraints lack approved supporting claim IDs: {invalid_constraint_sources}",
            })
        unknown_claims = sorted({
            claim_id for layer in plan.layers for decision in layer.decisions
            for claim_id in decision.cultural_claim_ids if claim_id not in claim_ids
        })
        if unknown_claims:
            issues.append({
                "severity": "blocking", "code": "unknown_cultural_claims",
                "message": f"The plan cites claims absent from the approved brief: {unknown_claims}",
            })
        if len(constraint_ids) != len(brief.constraints):
            issues.append({
                "severity": "blocking", "code": "duplicate_cultural_constraints",
                "message": "Cultural constraint IDs must be unique.",
            })
        return issues

    def review_screenplay(self, state: GraphState) -> GraphState:
        decision = interrupt({"gate": "screenplay", "project_id": state["project_id"]})
        if decision.get("action") != "approve":
            raise ValueError("Adapted screenplay must be approved to continue")
        self.repository.set_stage(state["project_id"], "building_visual_manifest")
        return {"adapted_revision": decision.get("revision_id") or state["adapted_revision"], "stage": "building_visual_manifest"}

    def visual_manifest(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        existing = self.repository.latest_revision(project_id, "visual_manifest")
        if existing:
            revision = existing
        else:
            revision = self._build_visual_manifest(project_id)
        payload = {"gate": "visuals", "message": "Approve canonical appearance and scene prompts before image generation.", "visual_revision": revision["id"]}
        self.repository.set_stage(project_id, "visual_review", interrupt=payload)
        return {"visual_revision": revision["id"], "stage": "visual_review"}

    def regenerate_visual_manifest(self, project_id: str) -> dict[str, Any]:
        project = self.repository.get_project(project_id)
        if project["stage"] != "visual_review":
            raise ValueError("Visual prompts can be regenerated only at the visual review gate")
        revision = self._build_visual_manifest(project_id)
        self.repository.set_stage(project_id, "visual_review", interrupt={
            "gate": "visuals", "message": "Review regenerated canonical appearance and scene prompts.",
            "visual_revision": revision["id"],
        })
        return revision

    def _build_visual_manifest(self, project_id: str) -> dict[str, Any]:
        source_revision = self.repository.latest_revision(project_id, "extraction")
        adapted_revision = self.repository.latest_revision(project_id, "adapted_screenplay")
        plan_revision = self.repository.latest_revision(project_id, "adaptation_plan")
        brief_revision = self.repository.latest_revision(project_id, "cultural_brief")
        if not all((source_revision, adapted_revision, plan_revision, brief_revision)):
            raise ValueError("Extraction, cultural brief, adaptation plan and adapted screenplay are required")
        source = SourceScreenplay.model_validate(source_revision["payload"])
        adapted = AdaptedScreenplay.model_validate(adapted_revision["payload"])
        plan = AdaptationPlan.model_validate(plan_revision["payload"])
        brief = CulturalBrief.model_validate(brief_revision["payload"])
        visual_context = self._culture_compiler(project_id, brief_revision).for_visual_manifest()

        appearance_scaffold, scene_scaffold = self._visual_scaffolds(project_id, source)
        trace_fields = {
            "profile_hash": visual_context.profile_hash,
            "brief_revision_id": brief_revision["id"],
            "cultural_claim_ids": [item["id"] for item in visual_context.claims],
            "cultural_constraint_ids": [item["id"] for item in visual_context.constraints],
        }
        appearance_scaffold = [{**item, **trace_fields} for item in appearance_scaffold]
        scene_scaffold = [{**item, **trace_fields} for item in scene_scaffold]
        visible_character_ids = [item["character_id"] for item in appearance_scaffold]

        relevant_plan = plan.model_dump(mode="json")
        relevant_plan["layers"] = [
            layer.model_dump(mode="json") for layer in plan.layers
            if layer.layer in {"non_verbal", "characters", "visual_world", "story_world", "cultural_precision"}
        ]
        source_payload = {
            "characters": [
                character.model_dump(mode="json") for character in source.characters
                if character.id in visible_character_ids
            ],
            "scenes": [scene.model_dump(mode="json") for scene in source.scenes],
            "production_elements": [element.model_dump(mode="json") for element in source.production_elements],
            "state_transitions": [transition.model_dump(mode="json") for transition in source.state_transitions],
        }
        current = self.repository.latest_revision(project_id, "visual_manifest")
        operation = f"build-visual-manifest-v{(current['version'] + 1) if current else 1}"
        if isinstance(self.ai.provider, MockProvider):
            appearances = []
            character_by_id = {character.id: character for character in source.characters}
            for scaffold in appearance_scaffold:
                character = character_by_id[scaffold["character_id"]]
                identity = f"{character.name}, {character.age_range or 'source-consistent age'}, {character.role or 'source role'}"
                costume = "Mock-only source-consistent everyday workwear; requires live visual planning."
                grooming = "Mock-only source-consistent grooming; requires live visual planning."
                appearances.append(AppearanceSpec(
                    **scaffold, identity_description=identity, costume_description=costume,
                    grooming_description=grooming,
                    prompt=(
                        f"Two-panel neutral production character reference with a close portrait and full-body view. "
                        f"Identity: {identity}. Costume: {costume}. Grooming: {grooming}. "
                        "Plain background, consistent proportions, natural posture, no caption, no decorative cultural invention."
                    ),
                ))
            scenes = [SceneVisualSpec(
                **scaffold, prompt=(
                    f"Mock cinematic keyframe for scene {index + 1}, preserving the supplied canonical appearance, location "
                    "and prop identifiers. Use a single readable dramatic moment, natural lighting and source-consistent "
                    "composition. This placeholder validates workflow structure only and requires live visual planning."
                ),
                negative_prompt="No unsupported cultural detail.",
            ) for index, scaffold in enumerate(scene_scaffold)]
            manifest = VisualManifest(appearances=appearances, scenes=scenes)
        else:
            manifest = self.ai.cached_structured(
                project_id, operation,
                visual_manifest_prompt(
                    source_payload, adapted.model_dump(mode="json"), relevant_plan, visual_context,
                    appearance_scaffold, scene_scaffold,
                ),
                VisualManifest,
            )
        for appearance in manifest.appearances:
            if VISUAL_STYLE_LOCK.lower() not in appearance.prompt.lower():
                appearance.prompt = VISUAL_STYLE_LOCK + "\n" + appearance.prompt.lstrip()
            required_lines = (
                f"Canonical identity: {appearance.identity_description}",
                f"Approved costume: {appearance.costume_description}",
                f"Approved grooming: {appearance.grooming_description}",
            )
            missing_lines = [line for line in required_lines if line.split(": ", 1)[1].lower() not in appearance.prompt.lower()]
            if missing_lines:
                appearance.prompt = appearance.prompt.rstrip() + "\n" + "\n".join(missing_lines)
        for scene in manifest.scenes:
            if VISUAL_STYLE_LOCK.lower() not in scene.prompt.lower():
                scene.prompt = VISUAL_STYLE_LOCK + "\n" + scene.prompt.lstrip()
            style_negative = "No illustration, cartoon, animation, anime, comic, painting, vector art or 3D-rendered look."
            if style_negative.lower() not in scene.negative_prompt.lower():
                scene.negative_prompt = scene.negative_prompt.rstrip(" ;") + "; " + style_negative
        scaffold_by_scene = {item["scene_id"]: item for item in scene_scaffold}
        for scene in manifest.scenes:
            scaffold = scaffold_by_scene[scene.scene_id]
            # Exact set identity is deterministic application data, never a model decision.
            scene.set_id = scaffold["set_id"]
            scene.sub_location = scaffold["sub_location"]
            scene.profile_hash = visual_context.profile_hash
            scene.brief_revision_id = brief_revision["id"]
            scene.cultural_claim_ids = scaffold["cultural_claim_ids"]
            scene.cultural_constraint_ids = scaffold["cultural_constraint_ids"]
        for appearance in manifest.appearances:
            appearance.profile_hash = visual_context.profile_hash
            appearance.brief_revision_id = brief_revision["id"]
            appearance.cultural_claim_ids = trace_fields["cultural_claim_ids"]
            appearance.cultural_constraint_ids = trace_fields["cultural_constraint_ids"]
        set_rows: dict[str, dict[str, Any]] = {}
        for scaffold in scene_scaffold:
            row = set_rows.setdefault(scaffold["set_id"], {
                "id": scaffold["set_id"], "location_id": scaffold["location_id"],
                "name": scaffold["sub_location"], "scene_ids": [],
                "profile_hash": visual_context.profile_hash, "brief_revision_id": brief_revision["id"],
            })
            row["scene_ids"].append(scaffold["scene_id"])
        manifest.sets = [SetVisualSpec.model_validate(item) for item in set_rows.values()]
        self._validate_visual_manifest(manifest, appearance_scaffold, scene_scaffold)
        return self.repository.create_revision(
            project_id, "visual_manifest", manifest, current["id"] if current else None,
        )

    def validate_visual_manifest_revision(self, project_id: str, revision_id: str) -> None:
        revision = self.repository.get_revision(revision_id)
        if revision["project_id"] != project_id or revision["kind"] != "visual_manifest":
            raise ValueError("The selected revision is not this project's visual manifest")
        source_revision = self.repository.latest_revision(project_id, "extraction")
        if not source_revision:
            raise ValueError("Extraction is required before visual approval")
        source = SourceScreenplay.model_validate(source_revision["payload"])
        appearance_scaffold, scene_scaffold = self._visual_scaffolds(project_id, source)
        brief_revision = self.repository.latest_revision(project_id, "cultural_brief")
        if not brief_revision:
            raise ValueError("An approved cultural brief is required before visual approval")
        context = self._culture_compiler(project_id, brief_revision).for_visual_manifest()
        trace_fields = {
            "profile_hash": context.profile_hash,
            "brief_revision_id": brief_revision["id"],
            "cultural_claim_ids": [item["id"] for item in context.claims],
            "cultural_constraint_ids": [item["id"] for item in context.constraints],
        }
        appearance_scaffold = [{**item, **trace_fields} for item in appearance_scaffold]
        scene_scaffold = [{**item, **trace_fields} for item in scene_scaffold]
        manifest = VisualManifest.model_validate(revision["payload"])
        scaffold_by_scene = {item["scene_id"]: item for item in scene_scaffold}
        for scene in manifest.scenes:
            scaffold = scaffold_by_scene[scene.scene_id]
            scene.set_id = scene.set_id or scaffold["set_id"]
            scene.sub_location = scene.sub_location or scaffold["sub_location"]
        self._validate_visual_manifest(manifest, appearance_scaffold, scene_scaffold)

    @staticmethod
    def _visual_scaffolds(
        project_id: str, source: SourceScreenplay,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        visible_character_ids = list(dict.fromkeys(
            character_id for scene in source.scenes for character_id in scene.character_ids
        ))
        appearance_scaffold = [{
            "id": stable_id(project_id, "appearance", character_id),
            "character_id": character_id,
            "scene_ids": [scene.id for scene in source.scenes if character_id in scene.character_ids],
        } for character_id in visible_character_ids]
        scene_scaffold = []
        for scene in source.scenes:
            sub_location = (scene.sub_location or scene.location or "Unknown set").strip()
            parent_key = scene.location_id or scene.location or "unknown-location"
            scene_scaffold.append({
                "id": stable_id(project_id, "scene_visual", scene.id),
                "scene_id": scene.id,
                "appearance_ids": [
                    stable_id(project_id, "appearance", character_id) for character_id in scene.character_ids
                ],
                "location_id": scene.location_id,
                "set_id": stable_id(project_id, "set", f"{parent_key}:{sub_location.casefold()}"),
                "sub_location": sub_location,
                "prop_ids": [element_id for element_id in scene.production_element_ids if element_id != scene.location_id],
            })
        return appearance_scaffold, scene_scaffold

    @staticmethod
    def _validate_visual_manifest(
        manifest: VisualManifest, appearance_scaffold: list[dict[str, Any]], scene_scaffold: list[dict[str, Any]],
    ) -> None:
        expected_appearances = [
            (
                item["id"], item["character_id"], item["scene_ids"], item.get("profile_hash", ""),
                item.get("brief_revision_id", ""), item.get("cultural_claim_ids", []),
                item.get("cultural_constraint_ids", []),
            ) for item in appearance_scaffold
        ]
        actual_appearances = [(
            item.id, item.character_id, item.scene_ids, item.profile_hash, item.brief_revision_id,
            item.cultural_claim_ids, item.cultural_constraint_ids,
        ) for item in manifest.appearances]
        if actual_appearances != expected_appearances:
            raise ValueError("Visual manifest changed, omitted or reordered canonical appearance IDs")
        expected_scenes = [(
            item["id"], item["scene_id"], item["appearance_ids"], item["location_id"],
            item["set_id"], item["sub_location"], item["prop_ids"], item.get("profile_hash", ""),
            item.get("brief_revision_id", ""), item.get("cultural_claim_ids", []),
            item.get("cultural_constraint_ids", []),
        ) for item in scene_scaffold]
        actual_scenes = [(
            item.id, item.scene_id, item.appearance_ids, item.location_id,
            item.set_id, item.sub_location, item.prop_ids, item.profile_hash, item.brief_revision_id,
            item.cultural_claim_ids, item.cultural_constraint_ids,
        ) for item in manifest.scenes]
        if actual_scenes != expected_scenes:
            raise ValueError("Visual manifest changed, omitted or reordered canonical scene dependencies")
        expected_set_ids = {item["set_id"] for item in scene_scaffold}
        if {item.id for item in manifest.sets} != expected_set_ids:
            raise ValueError("Visual manifest changed or omitted canonical set IDs")
        expected_profile_hash = appearance_scaffold[0].get("profile_hash", "") if appearance_scaffold else (
            scene_scaffold[0].get("profile_hash", "") if scene_scaffold else ""
        )
        expected_brief_revision = appearance_scaffold[0].get("brief_revision_id", "") if appearance_scaffold else (
            scene_scaffold[0].get("brief_revision_id", "") if scene_scaffold else ""
        )
        if any(
            item.profile_hash != expected_profile_hash or item.brief_revision_id != expected_brief_revision
            for item in manifest.sets
        ):
            raise ValueError("Visual set traceability does not match the approved culture context")
        placeholder_fragments = {
            "approved contemporary locality-appropriate everyday costume",
            "source-consistent grooming without inferred identity markers",
            "source-consistent age, face, body and grooming",
        }
        for appearance in manifest.appearances:
            combined = " ".join((
                appearance.identity_description, appearance.costume_description,
                appearance.grooming_description, appearance.prompt,
            )).lower()
            if any(fragment in combined for fragment in placeholder_fragments):
                raise ValueError(f"Appearance {appearance.id} still contains a generic visual placeholder")
            if min(
                len(appearance.identity_description), len(appearance.costume_description),
                len(appearance.grooming_description),
            ) < 30 or len(appearance.prompt) < 180:
                raise ValueError(f"Appearance {appearance.id} is not production-specific enough")
            prompt_lower = appearance.prompt.lower()
            for label, description in (
                ("identity", appearance.identity_description),
                ("costume", appearance.costume_description),
                ("grooming", appearance.grooming_description),
            ):
                if description.lower() not in prompt_lower:
                    raise ValueError(f"Appearance {appearance.id} prompt does not include its {label} description")
        if len({item.prompt for item in manifest.appearances}) != len(manifest.appearances):
            raise ValueError("Character appearance prompts must be distinct")
        for scene in manifest.scenes:
            if len(scene.prompt) < 180:
                raise ValueError(f"Scene prompt {scene.id} is not production-specific enough")
            negative_lower = scene.negative_prompt.lower()
            linguistic_fragments = ("dative", "future suffix", "copula", "pronoun", "नूं", "छै", "छो")
            if any(fragment in negative_lower for fragment in linguistic_fragments):
                raise ValueError(f"Scene prompt {scene.id} contains dialogue-only constraints")

    def review_visuals(self, state: GraphState) -> GraphState:
        decision = interrupt({"gate": "visuals", "project_id": state["project_id"]})
        if decision.get("action") != "approve":
            raise ValueError("Visual manifest must be approved to continue")
        self.repository.set_stage(state["project_id"], "preparing_character_assets")
        return {"visual_revision": decision.get("revision_id") or state["visual_revision"], "stage": "preparing_character_assets"}

    def prepare_character_assets(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        manifest = VisualManifest.model_validate(self.repository.get_revision(state["visual_revision"])["payload"])
        for appearance in manifest.appearances:
            self.repository.upsert_asset(
                project_id, "character_sheet", appearance.id, appearance.prompt,
                content_hash(appearance.model_dump(mode="json")),
            )
        payload = {"gate": "character_images", "message": "Generate and approve every canonical character sheet before keyframes."}
        self.repository.set_stage(project_id, "character_images_review", interrupt=payload)
        return {"stage": "character_images_review"}

    def review_character_images(self, state: GraphState) -> GraphState:
        decision = interrupt({"gate": "character_images", "project_id": state["project_id"]})
        if decision.get("action") != "approve":
            raise ValueError("Character images must be approved to continue")
        self.repository.set_stage(state["project_id"], "preparing_scene_assets")
        return {"stage": "preparing_scene_assets"}

    def prepare_scene_assets(self, state: GraphState) -> GraphState:
        project_id = state["project_id"]
        manifest = VisualManifest.model_validate(self.repository.get_revision(state["visual_revision"])["payload"])
        # Scene visuals are now compiled just in time.  Only Scene 1 exists until
        # its approved continuity snapshot unlocks the next scene.
        for scene in manifest.scenes[:1]:
            self.repository.upsert_asset(
                project_id, "scene_keyframe", scene.id,
                f"{scene.prompt}\nNegative constraints: {scene.negative_prompt}",
                content_hash(scene.model_dump(mode="json")), scene_id=scene.scene_id,
            )
        self.repository.set_stage(project_id, "scene_images_review", status="active")
        return {"stage": "scene_images_review"}

    def _normalize_ids(self, project_id: str, document: SourceScreenplay) -> SourceScreenplay:
        scene_map = {scene.id: stable_id(project_id, "scene", str(index)) for index, scene in enumerate(document.scenes, 1)}
        character_map = {character.id: stable_id(project_id, "character", character.name) for character in document.characters}
        element_map = {element.id: stable_id(project_id, element.kind, element.name) for element in document.production_elements}
        block_map: dict[str, str] = {}
        for index, scene in enumerate(document.scenes, 1):
            old_scene_id = scene.id
            scene.id = scene_map[old_scene_id]
            scene.number = index
            scene.character_ids = [character_map.get(item, item) for item in scene.character_ids]
            scene.production_element_ids = [element_map.get(item, item) for item in scene.production_element_ids]
            scene.location_id = element_map.get(scene.location_id, scene.location_id)
            for block_index, block in enumerate(scene.blocks):
                old_block_id = block.id
                block.id = stable_id(project_id, "block", f"{index}:{block_index}")
                block_map[old_block_id] = block.id
                block.speaker_id = character_map.get(block.speaker_id, block.speaker_id)
        for character in document.characters:
            old_id = character.id
            character.id = character_map[old_id]
            character.scene_entrances = [scene_map.get(item, item) for item in character.scene_entrances]
            character.scene_exits = [scene_map.get(item, item) for item in character.scene_exits]
            for relation in character.relationships:
                relation.target_character_id = character_map.get(relation.target_character_id, relation.target_character_id)
        for element in document.production_elements:
            old_id = element.id
            element.id = element_map[old_id]
            element.scene_ids = [scene_map.get(item, item) for item in element.scene_ids]
            element.owner_character_id = character_map.get(element.owner_character_id, element.owner_character_id)
        for transition in document.state_transitions:
            transition.character_id = character_map.get(transition.character_id, transition.character_id)
            transition.scene_id = scene_map.get(transition.scene_id, transition.scene_id)
            transition.before.character_id = character_map.get(transition.before.character_id, transition.before.character_id)
            transition.before.scene_id = scene_map.get(transition.before.scene_id, transition.before.scene_id)
            transition.after.character_id = character_map.get(transition.after.character_id, transition.after.character_id)
            transition.after.scene_id = scene_map.get(transition.after.scene_id, transition.after.scene_id)
        for index, event in enumerate(document.continuity_events, 1):
            event.id = stable_id(project_id, "continuity-event", f"{index}:{event.kind}")
            event.scene_id = scene_map.get(event.scene_id, event.scene_id)
            event.character_id = character_map.get(event.character_id, event.character_id)
            event.counterparty_character_id = character_map.get(
                event.counterparty_character_id, event.counterparty_character_id,
            )
            event.relationship_character_id = character_map.get(
                event.relationship_character_id, event.relationship_character_id,
            )
            event.prop_id = element_map.get(event.prop_id, event.prop_id)
            event.related_prop_id = element_map.get(event.related_prop_id, event.related_prop_id)
            event.evidence_block_ids = [block_map.get(item, item) for item in event.evidence_block_ids]
        if document.continuity_events:
            document.state_transitions, _ = build_state_ledger(document)
        document.story_contract.scene_purposes = {
            scene_map.get(scene_id, scene_id): purpose for scene_id, purpose in document.story_contract.scene_purposes.items()
        }
        return document
