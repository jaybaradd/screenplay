from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockType(StrEnum):
    action = "action"
    dialogue = "dialogue"
    transition = "transition"


class ContentBlock(StrictModel):
    id: str
    type: BlockType
    text: str
    speaker_id: str | None = None
    speaker_label: str | None = None
    source_start: int = 0
    source_end: int = 0


class SceneRecord(StrictModel):
    id: str
    number: int
    heading: str
    int_ext: str = "UNKNOWN"
    location_id: str | None = None
    location: str = "Unknown"
    sub_location: str | None = None
    time: str | None = None
    day_or_date: str | None = None
    weather: str | None = None
    mood: str | None = None
    summary: str
    dramatic_purpose: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    character_ids: list[str] = Field(default_factory=list)
    production_element_ids: list[str] = Field(default_factory=list)


class Relationship(StrictModel):
    target_character_id: str
    label: str
    notes: str | None = None


class CharacterRecord(StrictModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    age_range: str | None = None
    role: str | None = None
    relationships: list[Relationship] = Field(default_factory=list)
    personality: list[str] = Field(default_factory=list)
    dialect_register: str | None = None
    emotional_state: str | None = None
    scene_entrances: list[str] = Field(default_factory=list)
    scene_exits: list[str] = Field(default_factory=list)


class ProductionElement(StrictModel):
    id: str
    kind: Literal[
        "location", "set", "costume", "grooming", "jewellery", "prop", "food",
        "vehicle", "animal", "extra", "ritual", "gesture", "sound"
    ]
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    scene_ids: list[str] = Field(default_factory=list)
    owner_character_id: str | None = None


class StateSnapshot(StrictModel):
    character_id: str
    scene_id: str
    costume_id: str | None = None
    carries: list[str] = Field(default_factory=list)
    knows: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    relationship_state: dict[str, str] = Field(default_factory=dict)
    emotion: str | None = None


class StateTransition(StrictModel):
    character_id: str
    scene_id: str
    before: StateSnapshot
    events: list[str] = Field(default_factory=list)
    after: StateSnapshot


class ContinuityEvent(StrictModel):
    id: str
    scene_id: str
    character_id: str
    kind: Literal[
        "no_change", "first_observed_prop", "acquire_prop", "release_prop",
        "transfer_prop", "derive_prop", "learn_fact", "forget_fact",
        "costume_change", "injury", "recovery", "emotion_change", "relationship_change",
    ]
    timing: Literal["before_scene", "during_scene", "first_observed"] = "during_scene"
    prop_id: str | None = None
    related_prop_id: str | None = None
    counterparty_character_id: str | None = None
    fact: str | None = None
    value: str | None = None
    relationship_character_id: str | None = None
    evidence_block_ids: list[str] = Field(default_factory=list)
    description: str
    confidence: Literal["high", "medium", "low"] = "medium"


class ContinuityEventExtraction(StrictModel):
    events: list[ContinuityEvent] = Field(default_factory=list)
    new_production_elements: list[ProductionElement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StoryContract(StrictModel):
    central_dramatic_purpose: str
    relationship_invariants: list[str] = Field(default_factory=list)
    scene_purposes: dict[str, str] = Field(default_factory=dict)
    plot_invariants: list[str] = Field(default_factory=list)
    emotional_arc: list[str] = Field(default_factory=list)
    prohibited_changes: list[str] = Field(default_factory=list)


class SourceScreenplay(StrictModel):
    title: str
    detected_language: str
    original_text: str
    scenes: list[SceneRecord]
    characters: list[CharacterRecord]
    production_elements: list[ProductionElement]
    continuity_events: list[ContinuityEvent] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    story_contract: StoryContract
    extraction_warnings: list[str] = Field(default_factory=list)


class CulturalClaim(StrictModel):
    id: str
    claim: str
    scope: str
    time_period: str
    source_url: str
    origin: Literal["grounded_research", "reviewed_profile_evidence", "mock"]
    confidence: Literal["high", "medium", "low"]
    layers: list[str]
    uncertainty: str | None = None
    prohibited_extrapolations: list[str] = Field(default_factory=list)


class LanguageFeature(StrictModel):
    id: str
    category: Literal[
        "grammar", "pronoun", "honorific", "kinship", "particle", "idiom",
        "lexicon", "code_switching", "rhythm",
    ]
    written_form: str
    transliteration: str | None = None
    meaning_or_function: str
    usage_context: str
    speaker_constraints: list[str] = Field(default_factory=list)
    source_url: str
    confidence: Literal["high", "medium", "low"]
    prohibited_uses: list[str] = Field(default_factory=list)


class LanguageGuide(StrictModel):
    target_variety: str
    writing_script: str
    features: list[LanguageFeature] = Field(default_factory=list)
    register_rules: list[str] = Field(default_factory=list)
    code_switching_rules: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)


class CulturalConstraint(StrictModel):
    id: str
    text: str
    layers: list[str]
    kind: Literal["language_boundary", "visual_boundary", "research_boundary", "identity_boundary"]
    origin: Literal["grounded_research", "reviewed_profile_evidence", "profile_policy"]
    source_claim_ids: list[str] = Field(default_factory=list)


class CulturalBrief(StrictModel):
    culture_id: str
    profile_version: str
    profile_hash: str
    culture: str
    locality: str
    setting: str
    period: str
    output_script: str
    claims: list[CulturalClaim]
    language_guide: LanguageGuide | None = None
    constraints: list[CulturalConstraint]
    open_questions: list[str]
    research_summary: str


class LayerDecision(StrictModel):
    id: str
    source_observations: list[str]
    proposed_changes: list[str]
    affected_scene_ids: list[str]
    affected_character_ids: list[str]
    preserved_invariants: list[str]
    cultural_claim_ids: list[str]
    language_feature_ids: list[str] = Field(default_factory=list)
    uncertainty: list[str]
    risks: list[str]
    do_not_change: list[str]
    negative_constraints: list[str]


class LayerPlan(StrictModel):
    layer: Literal["verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision"]
    objective: str
    decisions: list[LayerDecision]


class AdaptationPlan(StrictModel):
    layers: list[LayerPlan]
    synthesis: list[str]
    deeper_change_requests: list[str]
    story_preservation_notes: list[str]


class AdaptedBlock(StrictModel):
    id: str
    source_block_ids: list[str]
    type: BlockType
    speaker_id: str | None = None
    adapted_text: str
    adaptation_layer_ids: list[str]
    cultural_claim_ids: list[str]
    explanation: str
    confidence: Literal["high", "medium", "low"]
    changed_dimensions: list[str]


class AdaptedScene(StrictModel):
    id: str
    source_scene_id: str
    heading: str
    summary: str
    blocks: list[AdaptedBlock]


class AdaptedScreenplay(StrictModel):
    title: str
    output_script: str
    scenes: list[AdaptedScene]
    preservation_summary: str


class SceneLanguageAudit(StrictModel):
    scene_id: str
    dialogue_block_ids: list[str] = Field(default_factory=list)
    supported_target_variety_block_ids: list[str] = Field(default_factory=list)
    unsupported_fallback_language_block_ids: list[str] = Field(default_factory=list)
    mixed_or_unapproved_variety_block_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LanguageAudit(StrictModel):
    target_variety: str
    scenes: list[SceneLanguageAudit]
    passed: bool
    summary: str


class ContinuityIssue(StrictModel):
    id: str
    severity: Literal["blocking", "warning", "info"]
    code: str
    entity_id: str | None = None
    expected: str | None = None
    actual: str | None = None
    affected_scene_ids: list[str]
    message: str
    suggested_resolution: str | None = None
    added_items: list[str] = Field(default_factory=list)
    removed_items: list[str] = Field(default_factory=list)
    evidence_block_ids: list[str] = Field(default_factory=list)
    probable_cause: str | None = None


class AppearanceSpec(StrictModel):
    id: str
    character_id: str
    costume_id: str | None = None
    scene_ids: list[str]
    identity_description: str
    costume_description: str
    grooming_description: str
    prompt: str
    cultural_claim_ids: list[str] = Field(default_factory=list)
    cultural_constraint_ids: list[str] = Field(default_factory=list)
    profile_hash: str = ""
    brief_revision_id: str = ""


class SceneVisualSpec(StrictModel):
    id: str
    scene_id: str
    appearance_ids: list[str]
    location_id: str | None = None
    # A location is the broader property/building; a set is the exact shootable
    # sub-location.  The optional default keeps older saved manifests readable.
    set_id: str | None = None
    sub_location: str | None = None
    prop_ids: list[str]
    prompt: str
    negative_prompt: str
    cultural_claim_ids: list[str] = Field(default_factory=list)
    cultural_constraint_ids: list[str] = Field(default_factory=list)
    profile_hash: str = ""
    brief_revision_id: str = ""


class SetVisualSpec(StrictModel):
    id: str
    location_id: str | None = None
    name: str
    scene_ids: list[str]
    profile_hash: str = ""
    brief_revision_id: str = ""


class VisualManifest(StrictModel):
    appearances: list[AppearanceSpec]
    scenes: list[SceneVisualSpec]
    sets: list[SetVisualSpec] = Field(default_factory=list)


class SetContinuitySnapshot(StrictModel):
    set_id: str
    location_id: str | None = None
    name: str
    reference_asset_id: str
    source_scene_id: str
    geometry_notes: str = ""
    fixed_elements: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    adjacency_notes: list[str] = Field(default_factory=list)
    mutable_elements: list[str] = Field(default_factory=list)
    approved_at: str


class SceneContinuitySnapshot(StrictModel):
    scene_id: str
    scene_number: int
    set_id: str
    asset_id: str
    appearance_ids: list[str]
    costume_ids: list[str] = Field(default_factory=list)
    prop_ids: list[str] = Field(default_factory=list)
    character_state_notes: list[str] = Field(default_factory=list)
    prop_state_notes: list[str] = Field(default_factory=list)
    approved_at: str


class VisualContinuityLedger(StrictModel):
    sets: list[SetContinuitySnapshot] = Field(default_factory=list)
    scenes: list[SceneContinuitySnapshot] = Field(default_factory=list)


class VisualVerificationIssue(StrictModel):
    severity: Literal["blocking", "warning"]
    dimension: Literal[
        "face", "apparent_age", "body", "grooming", "costume", "prop", "location",
        "composition", "visual_style", "set_geometry", "spatial_layout", "continuity", "other",
    ]
    expected: str
    observed: str
    message: str


class VisualVerification(StrictModel):
    asset_id: str
    dependency_hash: str
    passed: bool
    issues: list[VisualVerificationIssue] = Field(default_factory=list)
    summary: str


class CorrectionPatch(StrictModel):
    target_kind: Literal["adapted_block", "scene", "character", "plan_decision", "visual_prompt"]
    target_id: str
    precondition_hash: str
    operation: Literal["replace_text", "replace_field"]
    field: str
    new_value: Any
    explanation: str
    affected_dependencies: list[str]


class CorrectionContent(StrictModel):
    replacement_text: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ProjectCreate(StrictModel):
    title: str = "Untitled screenplay"
    source_text: str = Field(min_length=100, max_length=40_000)
    culture_id: str
    locality: str
    setting: str
    period: str
    output_script: str


class ApprovalRequest(StrictModel):
    revision_id: str | None = None
    override_reason: str | None = None


class SceneVisualApprovalRequest(StrictModel):
    geometry_notes: str = ""
    fixed_elements: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    adjacency_notes: list[str] = Field(default_factory=list)
    mutable_elements: list[str] = Field(default_factory=list)
    character_state_notes: list[str] = Field(default_factory=list)
    prop_state_notes: list[str] = Field(default_factory=list)
    promote_as_set_reference: bool = False
    override_verification: bool = False
    override_reason: str | None = Field(default=None, max_length=1000)
    override_reference_gate: bool = False
    reference_override_reason: str | None = Field(default=None, max_length=1000)


class AssetPromptRevisionRequest(StrictModel):
    prompt: str = Field(min_length=30, max_length=20_000)
    expected_dependency_hash: str


class CorrectionRequest(StrictModel):
    target_block_id: str
    instruction: str = Field(min_length=3, max_length=2000)
    expected_hash: str


class AssetCorrectionRequest(StrictModel):
    instruction: str = Field(min_length=3, max_length=2000)
    style_reference_asset_id: str | None = None


class MergeRequest(StrictModel):
    record_kind: Literal["character", "production_element"]
    primary_id: str
    duplicate_ids: list[str]


class RecordPatch(StrictModel):
    document_kind: str
    field_path: str
    value: Any
    expected_hash: str | None = None


class AssetStatus(StrEnum):
    pending = "pending"
    generated = "generated"
    approved = "approved"
    failed = "failed"
    invalidated = "invalidated"


class ProjectView(StrictModel):
    id: str
    title: str
    culture_id: str
    culture_display_name: str
    profile_version: str
    profile_hash: str
    approved_brief_revision_id: str | None = None
    locality: str
    setting: str
    period: str
    output_script: str
    stage: str
    status: str
    created_at: str
    updated_at: str
    revisions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    interrupt: dict[str, Any] | None = None
