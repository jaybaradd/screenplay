from __future__ import annotations

from typing import Any

from backend.cultures.models import CultureProfileSnapshot, CultureRuntimeContext, TargetSelection


class CultureContextCompiler:
    """Compile the smallest approved culture context required by each model stage."""

    def __init__(
        self,
        snapshot: CultureProfileSnapshot,
        selection: TargetSelection,
        brief: dict[str, Any] | None = None,
        *,
        brief_revision_id: str | None = None,
        brief_hash: str | None = None,
    ):
        self.snapshot = snapshot
        self.selection = selection
        self.brief = brief or {}
        self.brief_revision_id = brief_revision_id
        self.brief_hash = brief_hash

    def _compile(
        self,
        layers: set[str] | None,
        *,
        stage: str,
        scope_id: str | None = None,
        include_language: bool,
        include_visual: bool,
        include_research_policy: bool = False,
        include_reviewed_evidence: bool = False,
    ) -> CultureRuntimeContext:
        profile = self.snapshot.profile
        script = next(item for item in profile.supported_scripts if item.id == self.selection.output_script)
        claims = [
            item for item in self.brief.get("claims", [])
            if layers is None or layers.intersection(item.get("layers", []))
        ]
        def allowed_constraint(item: dict[str, Any]) -> bool:
            if layers is not None and not layers.intersection(item.get("layers", [])):
                return False
            kind = item.get("kind")
            if kind == "language_boundary":
                return include_language
            if kind in {"visual_boundary", "identity_boundary"}:
                return include_visual
            if kind == "research_boundary":
                return include_research_policy
            return False

        brief_constraints = [
            item for item in self.brief.get("constraints", []) if allowed_constraint(item)
        ]
        profile_constraints = [
            {**item.model_dump(mode="json"), "origin": "profile_policy"}
            for item in profile.constraints
            if allowed_constraint(item.model_dump(mode="json"))
        ]
        constraints_by_id = {item["id"]: item for item in brief_constraints}
        # The approved brief intentionally contains profile policies for review,
        # while the immutable snapshot remains authoritative at runtime.
        constraints_by_id.update({item["id"]: item for item in profile_constraints})
        return CultureRuntimeContext(
            stage=stage,
            scope_id=scope_id,
            culture_id=profile.culture_id,
            profile_version=profile.version,
            profile_hash=self.snapshot.profile_hash,
            display_name=profile.display_name,
            language_name=profile.language_name,
            target_variety=profile.target_variety,
            locality=self.selection.locality,
            setting=self.selection.setting,
            period=self.selection.period,
            output_script=script.id,
            output_script_name=script.display_name,
            confusable_languages=profile.commonly_confused_languages if include_language else [],
            confusable_visual_traditions=profile.commonly_confused_visual_traditions if include_visual else [],
            required_research_dimensions=profile.required_research_dimensions if include_research_policy else [],
            source_policy=profile.source_policy if include_research_policy else None,
            language_policy=profile.language_policy if include_language else None,
            visual_policy=profile.visual_policy if include_visual else None,
            research_guidance=profile.research_guidance if include_research_policy else [],
            claims=claims,
            constraints=list(constraints_by_id.values()),
            language_guide=self.brief.get("language_guide") if include_language else None,
            reviewed_evidence=(
                self.snapshot.reviewed_evidence if include_reviewed_evidence
                else self.snapshot.reviewed_evidence.model_copy(update={"claims": [], "language_features": []})
            ),
            brief_revision_id=self.brief_revision_id,
            brief_hash=self.brief_hash,
        )

    def for_research(self) -> CultureRuntimeContext:
        return self._compile(
            None, stage="research", include_language=True, include_visual=True,
            include_research_policy=True, include_reviewed_evidence=True,
        )

    def for_layer(self, layer: str) -> CultureRuntimeContext:
        include_language = layer in {"verbal", "cultural_precision"}
        include_visual = layer in {
            "non_verbal", "characters", "visual_world", "story_world", "cultural_precision",
        }
        return self._compile(
            {layer}, stage="layer", scope_id=layer,
            include_language=include_language, include_visual=include_visual,
        )

    def for_scene_adaptation(
        self, scene_id: str, claim_ids: set[str] | None = None,
    ) -> CultureRuntimeContext:
        context = self._compile(
            None, stage="scene_adaptation", scope_id=scene_id,
            include_language=True, include_visual=True,
        )
        if claim_ids is not None:
            context.claims = [item for item in context.claims if item.get("id") in claim_ids]
        return context

    def for_language_audit(self) -> CultureRuntimeContext:
        return self._compile(
            {"verbal", "cultural_precision"}, stage="language_audit",
            include_language=True, include_visual=False, include_reviewed_evidence=True,
        )

    def for_visual_manifest(self) -> CultureRuntimeContext:
        return self._compile(
            {"non_verbal", "characters", "visual_world", "story_world", "cultural_precision"},
            stage="visual_manifest", include_language=False, include_visual=True,
        )

    def for_visual_verification(self, scene_id: str) -> CultureRuntimeContext:
        return self._compile(
            {"non_verbal", "characters", "visual_world", "story_world", "cultural_precision"},
            stage="visual_verification", scope_id=scene_id,
            include_language=False, include_visual=True,
        )

    def for_correction(self, target_id: str) -> CultureRuntimeContext:
        return self._compile(
            None, stage="correction", scope_id=target_id,
            include_language=True, include_visual=True,
        )


def compiler_for_project(repository, project_id: str, brief_revision: dict[str, Any] | None = None) -> CultureContextCompiler:
    project = repository.get_project(project_id)
    snapshot_revision = repository.latest_revision(project_id, "culture_profile_snapshot")
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
