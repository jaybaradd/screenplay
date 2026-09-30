from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


PROFILE_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
SEMVER_PATTERN = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
LAYER_NAMES = {
    "verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision",
}


class CultureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScriptSpec(CultureModel):
    id: str = Field(min_length=1, pattern=PROFILE_ID_PATTERN)
    display_name: str = Field(min_length=1)
    font_path: str = Field(min_length=1)
    html_lang: str = Field(min_length=1)


class PeriodSpec(CultureModel):
    id: str = Field(min_length=1, pattern=PROFILE_ID_PATTERN)
    display_name: str = Field(min_length=1)


class SourcePolicy(CultureModel):
    preferred: list[str] = Field(min_length=1)
    discouraged: list[str]
    locality_specificity_rule: str = Field(min_length=1)


class LanguageEvidencePolicy(CultureModel):
    minimum_supported_features: int = Field(ge=1)
    require_source_per_feature: bool = True
    fallback_language: str | None = None
    fallback_requires_register_rule: bool = True
    required_feature_categories: list[Literal[
        "grammar", "pronoun", "honorific", "kinship", "particle", "idiom",
        "lexicon", "code_switching", "rhythm",
    ]] = Field(default_factory=list)


class CulturalPolicyConstraint(CultureModel):
    id: str
    text: str
    layers: list[Literal[
        "verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision",
    ]]
    kind: Literal["language_boundary", "visual_boundary", "research_boundary", "identity_boundary"]


class VisualPolicy(CultureModel):
    require_locality_evidence: bool = True
    require_period_evidence: bool = True
    prohibit_unapproved_festival_styling: bool = True


class CultureProfile(CultureModel):
    culture_id: str = Field(min_length=1, pattern=PROFILE_ID_PATTERN)
    version: str = Field(min_length=1, pattern=SEMVER_PATTERN)
    display_name: str = Field(min_length=1)
    production_enabled: bool = False
    language_name: str = Field(min_length=1)
    target_variety: str = Field(min_length=1)
    geographic_scope: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    supported_scripts: list[ScriptSpec] = Field(min_length=1)
    supported_settings: list[str] = Field(min_length=1)
    supported_periods: list[PeriodSpec] = Field(min_length=1)
    commonly_confused_languages: list[str] = Field(default_factory=list)
    commonly_confused_visual_traditions: list[str] = Field(default_factory=list)
    required_research_dimensions: list[str] = Field(min_length=1)
    research_guidance: list[str] = Field(default_factory=list)
    source_policy: SourcePolicy
    language_policy: LanguageEvidencePolicy
    visual_policy: VisualPolicy
    constraints: list[CulturalPolicyConstraint] = Field(default_factory=list)

    @field_validator("supported_settings")
    @classmethod
    def validate_settings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or not re.fullmatch(PROFILE_ID_PATTERN, item) for item in value):
            raise ValueError("Supported settings must use non-empty lowercase identifiers")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "CultureProfile":
        duplicate_groups = {
            "script IDs": [item.id for item in self.supported_scripts],
            "period IDs": [item.id for item in self.supported_periods],
            "settings": self.supported_settings,
            "constraint IDs": [item.id for item in self.constraints],
            "research dimensions": self.required_research_dimensions,
        }
        for label, values in duplicate_groups.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} are not allowed")
        unsupported_dimensions = set(self.required_research_dimensions) - LAYER_NAMES
        if unsupported_dimensions:
            raise ValueError(f"Unsupported research dimensions: {sorted(unsupported_dimensions)}")
        required_categories = self.language_policy.required_feature_categories
        if len(required_categories) != len(set(required_categories)):
            raise ValueError("Duplicate required language-feature categories are not allowed")
        return self


class ReviewedClaimEvidence(CultureModel):
    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    time_period: str = Field(min_length=1)
    source_url: HttpUrl
    confidence: Literal["high", "medium", "low"]
    permitted_layers: list[Literal[
        "verbal", "non_verbal", "characters", "visual_world", "story_world", "cultural_precision",
    ]] = Field(min_length=1)
    uncertainty: str | None = None
    prohibited_extrapolations: list[str] = Field(default_factory=list)


class ReviewedLanguageFeatureEvidence(CultureModel):
    id: str = Field(min_length=1)
    category: Literal[
        "grammar", "pronoun", "honorific", "kinship", "particle", "idiom",
        "lexicon", "code_switching", "rhythm",
    ]
    written_form: str = Field(min_length=1)
    transliteration: str | None = None
    meaning_or_function: str = Field(min_length=1)
    usage_context: str = Field(min_length=1)
    speaker_constraints: list[str] = Field(default_factory=list)
    source_url: HttpUrl
    confidence: Literal["high", "medium", "low"]
    prohibited_uses: list[str] = Field(default_factory=list)


class ReviewedEvidence(CultureModel):
    claims: list[ReviewedClaimEvidence] = Field(default_factory=list)
    language_features: list[ReviewedLanguageFeatureEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ReviewedEvidence":
        for label, values in {
            "reviewed claim IDs": [item.id for item in self.claims],
            "reviewed language-feature IDs": [item.id for item in self.language_features],
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} are not allowed")
        return self


class CultureProfileSnapshot(CultureModel):
    profile: CultureProfile
    reviewed_evidence: ReviewedEvidence
    profile_hash: str


class CultureProfileSummary(CultureModel):
    culture_id: str
    version: str
    display_name: str
    language_name: str
    target_variety: str
    geographic_scope: str
    limitations: list[str] = Field(default_factory=list)
    supported_scripts: list[ScriptSpec]
    supported_settings: list[str]
    supported_periods: list[PeriodSpec]


class TargetSelection(CultureModel):
    culture_id: str
    locality: str
    setting: str
    period: str
    output_script: str


class CultureRuntimeContext(CultureModel):
    stage: Literal[
        "research", "layer", "scene_adaptation", "language_audit",
        "visual_manifest", "visual_verification", "correction",
    ]
    scope_id: str | None = None
    culture_id: str
    profile_version: str
    profile_hash: str
    display_name: str
    language_name: str
    target_variety: str
    locality: str
    setting: str
    period: str
    output_script: str
    output_script_name: str
    confusable_languages: list[str]
    confusable_visual_traditions: list[str]
    required_research_dimensions: list[str] = Field(default_factory=list)
    source_policy: SourcePolicy | None = None
    language_policy: LanguageEvidencePolicy | None = None
    visual_policy: VisualPolicy | None = None
    research_guidance: list[str]
    claims: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    language_guide: dict[str, Any] | None = None
    reviewed_evidence: ReviewedEvidence = Field(default_factory=ReviewedEvidence)
    brief_revision_id: str | None = None
    brief_hash: str | None = None
