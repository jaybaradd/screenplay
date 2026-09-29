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
    adaptation_prompt, layer_prompt, synthesis_prompt,
)
from backend.providers import AIService, MockProvider, stable_id
from backend.schemas import (
    AdaptationPlan, AdaptedScene, AdaptedScreenplay, AppearanceSpec, CulturalBrief,
    LayerPlan, SceneVisualSpec, SourceScreenplay, VisualManifest,
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
        payload = {
            "gate": "plan", "message": "Review cultural brief, six adaptation layers and canonical records.",
            "plan_revision": plan_revision["id"], "cultural_revision": state["cultural_revision"],
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
                result = self.ai.cached_structured(
                    project_id, f"adapt-scene-{scene.number}",
                    adaptation_prompt(scene.model_dump(mode="json"), source_context, brief_payload, plan_payload),
                    AdaptedScene,
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
        mapped_blocks = {source_id for scene in adapted.scenes for block in scene.blocks for source_id in block.source_block_ids}
        missing = sorted(source_blocks - mapped_blocks)
        if missing:
            issues.append({"severity": "blocking", "code": "missing_source_blocks", "message": f"Unmapped source blocks: {missing}"})
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
            source = SourceScreenplay.model_validate(self.repository.get_revision(state["extraction_revision"])["payload"])
            brief = CulturalBrief.model_validate(self.repository.get_revision(state["cultural_revision"])["payload"])
            appearances: list[AppearanceSpec] = []
            for character in source.characters:
                scene_ids = [scene.id for scene in source.scenes if character.id in scene.character_ids]
                appearances.append(AppearanceSpec(
                    id=stable_id(project_id, "appearance", character.id), character_id=character.id,
                    scene_ids=scene_ids,
                    identity_description=f"{character.name}; {character.age_range or 'source-consistent apparent age'}; {character.role or 'source role'}",
                    costume_description="Approved contemporary locality-appropriate everyday costume; preserve across scenes unless the story records a change.",
                    grooming_description="Source-consistent grooming without inferred identity markers.",
                    prompt=(
                        f"Two-panel cinematic character reference sheet for {character.name}: close portrait and full-body costume view. "
                        f"Contemporary {brief.locality}, {brief.setting}; natural documentary realism. "
                        f"Identity: source-consistent age, face, body and grooming. Avoid: {', '.join(brief.negative_constraints)}"
                    ),
                ))
            scene_specs: list[SceneVisualSpec] = []
            for scene in source.scenes:
                appearance_ids = [stable_id(project_id, "appearance", character_id) for character_id in scene.character_ids]
                prop_ids = [element_id for element_id in scene.production_element_ids if element_id != scene.location_id]
                scene_specs.append(SceneVisualSpec(
                    id=stable_id(project_id, "scene_visual", scene.id), scene_id=scene.id,
                    appearance_ids=appearance_ids, location_id=scene.location_id, prop_ids=prop_ids,
                    prompt=(
                        f"Cinematic keyframe for scene {scene.number}: {scene.summary}. Location: {scene.location}; "
                        f"time: {scene.time or 'source-consistent'}; mood: {scene.mood or 'source-consistent'}. "
                        "Use supplied approved character references exactly; preserve costumes and visible props."
                    ),
                    negative_prompt="; ".join(brief.negative_constraints),
                ))
            manifest = VisualManifest(appearances=appearances, scenes=scene_specs)
            revision = self.repository.create_revision(project_id, "visual_manifest", manifest)
        payload = {"gate": "visuals", "message": "Approve canonical appearance and scene prompts before image generation.", "visual_revision": revision["id"]}
        self.repository.set_stage(project_id, "visual_review", interrupt=payload)
        return {"visual_revision": revision["id"], "stage": "visual_review"}

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
        for scene in manifest.scenes:
            self.repository.upsert_asset(
                project_id, "scene_keyframe", scene.id,
                f"{scene.prompt}\nNegative constraints: {scene.negative_prompt}",
                content_hash(scene.model_dump(mode="json")), scene_id=scene.scene_id,
            )
        self.repository.set_stage(project_id, "visuals_ready", status="ready")
        return {"stage": "visuals_ready"}

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
