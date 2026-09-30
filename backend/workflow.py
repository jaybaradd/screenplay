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
from backend.continuity import check_continuity
from backend.observability import observer
from backend.prompts import (
    CULTURAL_NORMALIZE_PROMPT, CULTURAL_RESEARCH_PROMPT, EXTRACTION_PROMPT,
    adaptation_prompt, adaptation_repair_prompt, dialect_audit_prompt, layer_prompt, synthesis_prompt,
    visual_manifest_prompt, VISUAL_STYLE_LOCK,
)
from backend.providers import AIService, MockProvider, stable_id
from backend.schemas import (
    AdaptationPlan, AdaptedScene, AdaptedScreenplay, AppearanceSpec, CulturalBrief,
    DialectAudit, LayerPlan, SceneRecord, SceneVisualSpec, SetVisualSpec, SourceScreenplay, VisualManifest,
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
            with observer.span(f"graph-node:{name}", project_id=state["project_id"], metadata={"node": name}):
                return node(state)
        return wrapped

    def config(self, project_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": project_id}}

    def start(self, project_id: str) -> dict[str, Any]:
        with observer.span("workflow-start", project_id=project_id):
            return self.graph.invoke(
                {"project_id": project_id, "layer_revision_ids": [], "stage": "created"},
                self.config(project_id),
            )

    def resume(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with observer.span("workflow-resume", project_id=project_id, metadata={"action": payload.get("action")}):
            return self.graph.invoke(Command(resume=payload), self.config(project_id))

    def retry_failed(self, project_id: str) -> dict[str, Any]:
        with observer.span("workflow-retry", project_id=project_id):
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
        project = self.repository.get_project(project_id)
        if isinstance(self.ai.provider, MockProvider):
            self.ai.provider.set_context(locality=project["locality"], setting=project["setting"], period=project["period"])
        research_prompt = CULTURAL_RESEARCH_PROMPT.format(**project)
        research = self.ai.cached_research(project_id, "cultural-grounding", research_prompt)
        normalize_prompt = CULTURAL_NORMALIZE_PROMPT.format(research=research)
        brief = self.ai.cached_structured(project_id, "normalize-cultural-brief", normalize_prompt, CulturalBrief)
        revision = self.repository.create_revision(project_id, "cultural_brief", brief)
        return {"cultural_revision": revision["id"], "layer_revision_ids": []}

    def _layer_node(self, layer: str):
        def node(state: GraphState) -> GraphState:
            project_id = state["project_id"]
            existing = self.repository.latest_revision(project_id, f"plan_{layer}")
            if existing:
                return {"layer_revision_ids": [existing["id"]]}
            source = self.repository.get_revision(state["extraction_revision"])["payload"]
            brief = self.repository.get_revision(state["cultural_revision"])["payload"]
            result = self.ai.cached_structured(project_id, f"plan-{layer}", layer_prompt(layer, source, brief), LayerPlan)
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
        brief = CulturalBrief.model_validate(self.repository.get_revision(state["cultural_revision"])["payload"])
        plan = AdaptationPlan.model_validate(plan_revision["payload"])
        plan_issues = self._plan_quality_issues(source, brief, plan)
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
            brief = CulturalBrief.model_validate(self.repository.get_revision(state["cultural_revision"])["payload"])
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
                prompt = adaptation_prompt(scene_payload, source_context, brief_payload, plan_payload)
                result = self.ai.cached_structured(
                    project_id, f"adapt-scene-{scene.number}-attempt-1", prompt, AdaptedScene,
                )
                errors = self._scene_mapping_errors(scene, result)
                for attempt in range(2, 4):
                    if not errors:
                        break
                    prompt = adaptation_repair_prompt(
                        scene_payload, source_context, brief_payload, plan_payload,
                        result.model_dump(mode="json"), errors,
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
        if not isinstance(self.ai.provider, MockProvider):
            brief = CulturalBrief.model_validate(self.repository.get_revision(state["cultural_revision"])["payload"])
            audit = self.ai.cached_structured(
                project_id, "audit-adapted-dialect",
                dialect_audit_prompt(adapted.model_dump(mode="json"), brief.model_dump(mode="json")),
                DialectAudit,
            )
            previous_audit = self.repository.latest_revision(project_id, "dialect_audit")
            self.repository.create_revision(
                project_id, "dialect_audit", audit,
                previous_audit["id"] if previous_audit else None,
            )
            audit_by_scene = {item.scene_id: item for item in audit.scenes}
            for scene in adapted.scenes:
                expected_dialogue_ids = {
                    block.id for block in scene.blocks if block.type.value == "dialogue"
                }
                scene_audit = audit_by_scene.get(scene.source_scene_id)
                if not scene_audit:
                    if expected_dialogue_ids:
                        issues.append({
                            "severity": "blocking", "code": "dialect_audit_incomplete",
                            "message": f"Dialect audit omitted speaking scene {scene.source_scene_id}.",
                        })
                    continue
                declared = set(scene_audit.dialogue_block_ids)
                classified_sequence = [
                    *scene_audit.supported_mewari_block_ids,
                    *scene_audit.generic_hindi_block_ids,
                    *scene_audit.unapproved_or_mixed_block_ids,
                ]
                classified = set(classified_sequence)
                missing_audit_blocks = sorted(expected_dialogue_ids - classified)
                unknown_audit_blocks = sorted((declared | classified) - expected_dialogue_ids)
                duplicate_classifications = sorted({
                    block_id for block_id in classified_sequence if classified_sequence.count(block_id) > 1
                })
                if declared != expected_dialogue_ids or missing_audit_blocks or unknown_audit_blocks or duplicate_classifications:
                    issues.append({
                        "severity": "blocking", "code": "dialect_audit_incomplete",
                        "message": (
                            f"Dialect audit for scene {scene.source_scene_id} is incomplete or inconsistent. "
                            f"Missing={missing_audit_blocks}; unknown={unknown_audit_blocks}; "
                            f"multiply classified={duplicate_classifications}."
                        ),
                    })
            for scene_audit in audit.scenes:
                if scene_audit.generic_hindi_block_ids:
                    issues.append({
                        "severity": "blocking", "code": "generic_hindi_dialogue",
                        "message": (
                            f"Scene {scene_audit.scene_id} contains dialogue assessed as standard Hindi rather than "
                            f"Maidani Mewari: {scene_audit.generic_hindi_block_ids}"
                        ),
                    })
                if scene_audit.unapproved_or_mixed_block_ids:
                    issues.append({
                        "severity": "blocking", "code": "unapproved_dialect_mixing",
                        "message": (
                            f"Scene {scene_audit.scene_id} contains unsupported, mixed or potentially Marwari forms: "
                            f"{scene_audit.unapproved_or_mixed_block_ids}"
                        ),
                    })
            if not audit.passed and not any(
                issue["code"] in {"generic_hindi_dialogue", "unapproved_dialect_mixing"} for issue in issues
            ):
                issues.append({"severity": "blocking", "code": "dialect_audit_failed", "message": audit.summary})
        coverage = (len(mapped_blocks & source_blocks) / len(source_blocks)) if source_blocks else 1.0
        observer.score("story_anchor_coverage", coverage, project_id=project_id)
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
        source: SourceScreenplay, brief: CulturalBrief, plan: AdaptationPlan,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        guide = brief.dialect_guide
        if not guide or guide.target_variety.strip().lower() != "maidani mewari":
            issues.append({
                "severity": "blocking", "code": "missing_target_dialect_guide",
                "message": "The cultural brief does not contain a Maidani Mewari dialect guide.",
            })
            return issues
        if "devanagari" not in guide.writing_script.strip().lower():
            issues.append({
                "severity": "blocking", "code": "wrong_dialect_script",
                "message": f"Dialect guide script must be Devanagari, got {guide.writing_script}.",
            })
        usable_features = [feature for feature in guide.features if feature.confidence in {"high", "medium"}]
        if len(usable_features) < 6:
            issues.append({
                "severity": "blocking", "code": "insufficient_dialect_evidence",
                "message": (
                    "Fewer than six medium/high-confidence, source-backed Maidani Mewari language features are available; "
                    "a native-feeling adaptation cannot be grounded safely."
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
        plan_text = " ".join(
            change for decision in verbal.decisions for change in decision.proposed_changes
        ).lower()
        feature_markers = [feature.id.lower() for feature in guide.features] + [
            feature.devanagari_form.lower() for feature in guide.features
        ]
        if guide.features and not any(marker and marker in plan_text for marker in feature_markers):
            issues.append({
                "severity": "blocking", "code": "verbal_plan_not_grounded",
                "message": "The verbal plan does not cite or use any approved DialectFeature ID or Devanagari form.",
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

        appearance_scaffold, scene_scaffold = self._visual_scaffolds(project_id, source)
        visible_character_ids = [item["character_id"] for item in appearance_scaffold]

        relevant_plan = plan.model_dump(mode="json")
        relevant_plan["layers"] = [
            layer.model_dump(mode="json") for layer in plan.layers
            if layer.layer in {"non_verbal", "characters", "visual_world", "story_world", "cultural_precision"}
        ]
        visual_brief = brief.model_dump(mode="json")
        visual_brief.pop("dialect_guide", None)
        visual_brief["claims"] = [
            claim.model_dump(mode="json") for claim in brief.claims
            if any(layer in {"non_verbal", "characters", "visual_world", "story_world", "cultural_precision"} for layer in claim.layers)
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
                    source_payload, adapted.model_dump(mode="json"), relevant_plan, visual_brief,
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
        set_rows: dict[str, dict[str, Any]] = {}
        for scaffold in scene_scaffold:
            row = set_rows.setdefault(scaffold["set_id"], {
                "id": scaffold["set_id"], "location_id": scaffold["location_id"],
                "name": scaffold["sub_location"], "scene_ids": [],
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
            (item["id"], item["character_id"], item["scene_ids"]) for item in appearance_scaffold
        ]
        actual_appearances = [(item.id, item.character_id, item.scene_ids) for item in manifest.appearances]
        if actual_appearances != expected_appearances:
            raise ValueError("Visual manifest changed, omitted or reordered canonical appearance IDs")
        expected_scenes = [(
            item["id"], item["scene_id"], item["appearance_ids"], item["location_id"],
            item["set_id"], item["sub_location"], item["prop_ids"],
        ) for item in scene_scaffold]
        actual_scenes = [(
            item.id, item.scene_id, item.appearance_ids, item.location_id,
            item.set_id, item.sub_location, item.prop_ids,
        ) for item in manifest.scenes]
        if actual_scenes != expected_scenes:
            raise ValueError("Visual manifest changed, omitted or reordered canonical scene dependencies")
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
        for index, scene in enumerate(document.scenes, 1):
            old_scene_id = scene.id
            scene.id = scene_map[old_scene_id]
            scene.number = index
            scene.character_ids = [character_map.get(item, item) for item in scene.character_ids]
            scene.production_element_ids = [element_map.get(item, item) for item in scene.production_element_ids]
            scene.location_id = element_map.get(scene.location_id, scene.location_id)
            for block_index, block in enumerate(scene.blocks):
                block.id = stable_id(project_id, "block", f"{index}:{block_index}")
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
        document.story_contract.scene_purposes = {
            scene_map.get(scene_id, scene_id): purpose for scene_id, purpose in document.story_contract.scene_purposes.items()
        }
        return document
