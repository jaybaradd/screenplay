from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
import zipfile
from html import escape
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.continuity import check_continuity
from backend.prompts import correction_prompt, VISUAL_STYLE_LOCK
from backend.providers import AIService, MockProvider, stable_id
from backend.schemas import (
    AdaptedScreenplay, AssetCorrectionRequest, AssetStatus, CorrectionPatch, CorrectionRequest, MergeRequest,
    RecordPatch, SceneContinuitySnapshot, SceneVisualApprovalRequest, SetContinuitySnapshot,
    SourceScreenplay, VisualContinuityLedger, VisualManifest, utc_now,
)
from backend.storage import Repository, content_hash


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.strip("/").split("/") if part]
    cursor: Any = payload
    for part in parts[:-1]:
        cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
    final = parts[-1]
    if isinstance(cursor, list):
        cursor[int(final)] = value
    else:
        cursor[final] = value


class ProjectService:
    def __init__(self, repository: Repository, ai: AIService):
        self.repository = repository
        self.ai = ai

    def patch_record(self, project_id: str, patch: RecordPatch) -> dict[str, Any]:
        if patch.document_kind == "visual_manifest" and self.repository.get_project(project_id)["stage"] != "visual_review":
            raise ValueError("Visual prompts can be edited only at the visual review gate")
        current = self.repository.latest_revision(project_id, patch.document_kind)
        if not current:
            raise KeyError(f"No {patch.document_kind} document exists")
        if patch.expected_hash and patch.expected_hash != current["sha256"]:
            raise ValueError("The document changed after it was loaded; refresh before editing")
        if patch.field_path.strip("/"):
            payload = copy.deepcopy(current["payload"])
            _set_path(payload, patch.field_path, patch.value)
        else:
            payload = copy.deepcopy(patch.value)
        if patch.document_kind == "visual_manifest":
            payload = VisualManifest.model_validate(payload).model_dump(mode="json")
        revision = self.repository.create_revision(project_id, patch.document_kind, payload, current["id"])
        if patch.document_kind == "extraction":
            self._refresh_continuity(project_id, revision)
        return revision

    def merge_records(self, project_id: str, request: MergeRequest) -> dict[str, Any]:
        current = self.repository.latest_revision(project_id, "extraction")
        if not current:
            raise KeyError("Extraction does not exist")
        source = SourceScreenplay.model_validate(current["payload"])
        duplicate_ids = set(request.duplicate_ids)
        if request.primary_id in duplicate_ids:
            duplicate_ids.remove(request.primary_id)
        if request.record_kind == "character":
            records = {record.id: record for record in source.characters}
            primary = records.get(request.primary_id)
            if not primary or not duplicate_ids.issubset(records):
                raise ValueError("Unknown primary or duplicate character ID")
            for duplicate_id in duplicate_ids:
                duplicate = records[duplicate_id]
                primary.aliases = list(dict.fromkeys([*primary.aliases, duplicate.name, *duplicate.aliases]))
            source.characters = [record for record in source.characters if record.id not in duplicate_ids]
            for scene in source.scenes:
                scene.character_ids = list(dict.fromkeys(request.primary_id if item in duplicate_ids else item for item in scene.character_ids))
                for block in scene.blocks:
                    if block.speaker_id in duplicate_ids:
                        block.speaker_id = request.primary_id
            for transition in source.state_transitions:
                if transition.character_id in duplicate_ids:
                    transition.character_id = request.primary_id
                if transition.before.character_id in duplicate_ids:
                    transition.before.character_id = request.primary_id
                if transition.after.character_id in duplicate_ids:
                    transition.after.character_id = request.primary_id
        else:
            records = {record.id: record for record in source.production_elements}
            primary = records.get(request.primary_id)
            if not primary or not duplicate_ids.issubset(records):
                raise ValueError("Unknown primary or duplicate production element ID")
            for duplicate_id in duplicate_ids:
                duplicate = records[duplicate_id]
                primary.aliases = list(dict.fromkeys([*primary.aliases, duplicate.name, *duplicate.aliases]))
            source.production_elements = [record for record in source.production_elements if record.id not in duplicate_ids]
            for scene in source.scenes:
                scene.production_element_ids = list(dict.fromkeys(request.primary_id if item in duplicate_ids else item for item in scene.production_element_ids))
                if scene.location_id in duplicate_ids:
                    scene.location_id = request.primary_id
        revision = self.repository.create_revision(project_id, "extraction", source, current["id"])
        self._refresh_continuity(project_id, revision)
        return revision

    def _refresh_continuity(self, project_id: str, extraction_revision: dict[str, Any]) -> None:
        source = SourceScreenplay.model_validate(extraction_revision["payload"])
        issues = check_continuity(source)
        previous = self.repository.latest_revision(project_id, "continuity_issues")
        self.repository.create_revision(
            project_id, "continuity_issues",
            [issue.model_dump(mode="json") for issue in issues],
            previous["id"] if previous else None,
        )

    def correct_block(self, project_id: str, request: CorrectionRequest) -> dict[str, Any]:
        current = self.repository.latest_revision(project_id, "adapted_screenplay")
        if not current:
            raise KeyError("Adapted screenplay does not exist")
        screenplay = AdaptedScreenplay.model_validate(current["payload"])
        target = None
        target_scene = None
        neighbours: list[dict[str, Any]] = []
        for scene in screenplay.scenes:
            for index, block in enumerate(scene.blocks):
                if block.id == request.target_block_id:
                    target = block
                    target_scene = scene
                    neighbours = [item.model_dump(mode="json") for item in scene.blocks[max(0, index - 1):index + 2]]
                    break
        if not target or not target_scene:
            raise KeyError("Target block does not exist")
        target_payload = target.model_dump(mode="json")
        actual_hash = content_hash(target_payload)
        if request.expected_hash != actual_hash:
            raise ValueError("The selected block changed after it was loaded; refresh before correcting")
        if isinstance(self.ai.provider, MockProvider):
            patch = CorrectionPatch(
                target_kind="adapted_block", target_id=target.id, precondition_hash=actual_hash,
                operation="replace_text", field="adapted_text", new_value=request.instruction,
                explanation="Mock mode treats the correction instruction as the exact replacement text.",
                affected_dependencies=[] if target.type == "dialogue" else [target_scene.id],
            )
        else:
            extraction = self.repository.latest_revision(project_id, "extraction")
            brief = self.repository.latest_revision(project_id, "cultural_brief")
            context = {
                "neighbours": neighbours,
                "story_contract": extraction["payload"]["story_contract"] if extraction else {},
                "cultural_brief": brief["payload"] if brief else {},
            }
            patch = self.ai.cached_structured(
                project_id, f"correct-block-{target.id}", correction_prompt(target_payload, request.instruction, context), CorrectionPatch,
            )
        if patch.target_id != target.id or patch.precondition_hash != actual_hash or patch.field != "adapted_text":
            raise ValueError("Correction model attempted to modify an unapproved target")
        before_hashes = {
            block.id: content_hash(block.model_dump(mode="json"))
            for scene in screenplay.scenes for block in scene.blocks if block.id != target.id
        }
        target.adapted_text = str(patch.new_value)
        after_hashes = {
            block.id: content_hash(block.model_dump(mode="json"))
            for scene in screenplay.scenes for block in scene.blocks if block.id != target.id
        }
        if before_hashes != after_hashes:
            raise RuntimeError("Surgical edit isolation failed: an unrelated block changed")
        revision = self.repository.create_revision(project_id, "adapted_screenplay", screenplay, current["id"])
        ledger = self.repository.latest_revision(project_id, "change_ledger")
        entries = list(ledger["payload"]) if ledger else []
        entries.append({
            "revision_id": revision["id"], "target_id": target.id, "before_hash": actual_hash,
            "after_hash": content_hash(target.model_dump(mode="json")), "instruction": request.instruction,
            "explanation": patch.explanation, "unchanged_block_count": len(before_hashes),
        })
        self.repository.create_revision(project_id, "change_ledger", entries, ledger["id"] if ledger else None)
        if patch.affected_dependencies:
            manifest_revision = self.repository.latest_revision(project_id, "visual_manifest")
            if manifest_revision:
                manifest = VisualManifest.model_validate(manifest_revision["payload"])
                affected = [item.id for item in manifest.scenes if item.scene_id in patch.affected_dependencies]
                self.repository.invalidate_assets(project_id, affected)
        return revision

    def start_visual_continuity(self, project_id: str) -> dict[str, Any]:
        """Opt an older project into sequential scene approval without deleting assets."""
        project = self.repository.get_project(project_id)
        if project["stage"] not in {"visuals_ready", "scene_images_review"}:
            raise ValueError("Sequential visual continuity starts after character references are approved")
        current = self.repository.latest_revision(project_id, "visual_continuity")
        if not current:
            current = self.repository.create_revision(project_id, "visual_continuity", VisualContinuityLedger())
        self.repository.set_stage(project_id, "scene_images_review", status="active")
        return current

    def _visual_records(self, project_id: str) -> tuple[SourceScreenplay, VisualManifest]:
        extraction = self.repository.latest_revision(project_id, "extraction")
        manifest_revision = self.repository.latest_revision(project_id, "visual_manifest")
        if not extraction or not manifest_revision:
            raise ValueError("Extraction and visual manifest are required")
        source = SourceScreenplay.model_validate(extraction["payload"])
        manifest = VisualManifest.model_validate(manifest_revision["payload"])
        for appearance in manifest.appearances:
            appearance.costume_id = appearance.costume_id or stable_id(
                project_id, "costume", f"{appearance.id}:{content_hash(appearance.costume_description)[:16]}",
            )
        source_by_id = {scene.id: scene for scene in source.scenes}
        for spec in manifest.scenes:
            source_scene = source_by_id[spec.scene_id]
            sub_location = (spec.sub_location or source_scene.sub_location or source_scene.location or "Unknown set").strip()
            parent_key = spec.location_id or source_scene.location_id or source_scene.location or "unknown-location"
            spec.sub_location = sub_location
            spec.set_id = spec.set_id or stable_id(project_id, "set", f"{parent_key}:{sub_location.casefold()}")
        return source, manifest

    def _ledger(self, project_id: str) -> tuple[VisualContinuityLedger, dict[str, Any] | None]:
        revision = self.repository.latest_revision(project_id, "visual_continuity")
        return (
            VisualContinuityLedger.model_validate(revision["payload"]) if revision else VisualContinuityLedger(),
            revision,
        )

    def compile_scene_asset(self, project_id: str, scene_id: str) -> dict[str, Any]:
        """Compile a scene prompt from approved set, character and prior-scene state."""
        source, manifest = self._visual_records(project_id)
        source_index = {scene.id: index for index, scene in enumerate(source.scenes)}
        if scene_id not in source_index:
            raise KeyError("Scene does not exist")
        index = source_index[scene_id]
        spec = next(item for item in manifest.scenes if item.scene_id == scene_id)
        ledger, ledger_revision = self._ledger(project_id)
        approved_scene_ids = {item.scene_id for item in ledger.scenes}
        missing_prior = [scene.id for scene in source.scenes[:index] if scene.id not in approved_scene_ids]
        if missing_prior:
            raise ValueError("Approve every preceding scene before compiling this scene")

        set_snapshot = next((item for item in ledger.sets if item.set_id == spec.set_id), None)
        previous_snapshot = next(
            (item for item in ledger.scenes if index and item.scene_id == source.scenes[index - 1].id), None,
        )
        appearances = {item.id: item for item in manifest.appearances}
        identity_lines = []
        for appearance_id in spec.appearance_ids:
            appearance = appearances[appearance_id]
            identity_lines.append(
                f"- {appearance_id}: identity={appearance.identity_description}; "
                f"costume_id={appearance.costume_id or 'baseline'}; costume={appearance.costume_description}; "
                f"grooming={appearance.grooming_description}"
            )
        set_lines = [f"set_id={spec.set_id}; exact sub-location={spec.sub_location}"]
        if set_snapshot:
            set_lines.extend([
                f"geometry={set_snapshot.geometry_notes or 'preserve approved reference geometry'}",
                f"fixed elements={'; '.join(set_snapshot.fixed_elements) or 'preserve approved reference'}",
                f"materials={'; '.join(set_snapshot.materials) or 'preserve approved reference'}",
                f"palette={'; '.join(set_snapshot.palette) or 'preserve approved reference'}",
                f"adjacency={'; '.join(set_snapshot.adjacency_notes) or 'no additional adjacency claim'}",
            ])
        prior_lines = []
        if previous_snapshot:
            prior_lines = [
                f"previous scene={previous_snapshot.scene_id}",
                f"character state={'; '.join(previous_snapshot.character_state_notes) or 'preserve approved identities and costumes'}",
                f"prop state={'; '.join(previous_snapshot.prop_state_notes) or 'use the state ledger and current scene specification'}",
            ]
        compiled_prompt = (
            f"{VISUAL_STYLE_LOCK}\n\n"
            "BASE APPROVED SCENE SPECIFICATION:\n"
            f"{spec.prompt}\nNegative constraints: {spec.negative_prompt}\n\n"
            "CONTINUITY INVARIANTS — MUST NOT CHANGE:\n"
            + "\n".join(set_lines + identity_lines + prior_lines)
            + "\n\nPERMITTED SCENE DELTAS:\n"
            "Only action, blocking, expression, temporary props and lighting explicitly required by this scene may change. "
            "Do not redesign the set, face, body, apparent age, grooming or active costume.\n\n"
            "REFERENCE PRIORITY:\n"
            "Same-set approved reference controls architecture and spatial layout. Character sheets control identity and costume. "
            "The immediately preceding approved scene controls carried state only; it must not overwrite a different set."
        )
        all_assets = self.repository.list_assets(project_id)
        reference_rows: list[dict[str, str]] = []
        if set_snapshot:
            reference_rows.append({
                "asset_id": set_snapshot.reference_asset_id,
                "label": f"AUTHORITATIVE SAME-SET REFERENCE — {set_snapshot.name}; preserve geometry and fixed elements",
            })
        for appearance_id in spec.appearance_ids:
            reference = next((
                item for item in reversed(all_assets)
                if item["kind"] == "character_sheet" and item["canonical_id"] == appearance_id
                and item["status"] == AssetStatus.approved and item["path"]
            ), None)
            if reference:
                reference_rows.append({
                    "asset_id": reference["id"],
                    "label": f"IMMUTABLE CHARACTER/COSTUME REFERENCE — {appearance_id}",
                })
        if previous_snapshot and previous_snapshot.asset_id not in {row["asset_id"] for row in reference_rows}:
            reference_rows.append({
                "asset_id": previous_snapshot.asset_id,
                "label": "IMMEDIATE PRIOR SCENE — carry only character, costume and prop state; do not copy its set",
            })
        dependency_hash = content_hash({
            "base_spec": spec.model_dump(mode="json"),
            "ledger_hash": ledger_revision["sha256"] if ledger_revision else None,
            "references": reference_rows,
            "compiled_prompt": compiled_prompt,
        })
        asset = self.repository.upsert_asset(
            project_id, "scene_keyframe", spec.id, compiled_prompt, dependency_hash, scene_id=scene_id,
        )
        history = self.repository.latest_revision(project_id, "scene_prompt_compilations")
        records = list(history["payload"]) if history else []
        records.append({
            "asset_id": asset["id"], "scene_id": scene_id, "set_id": spec.set_id,
            "reference_assets": reference_rows, "dependency_hash": dependency_hash, "created_at": utc_now(),
        })
        self.repository.create_revision(
            project_id, "scene_prompt_compilations", records, history["id"] if history else None,
        )
        self.repository.set_stage(project_id, "scene_images_review", status="active")
        return asset

    def approve_scene_asset(self, asset_id: str, request: SceneVisualApprovalRequest) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        if asset["kind"] != "scene_keyframe" or not asset["scene_id"]:
            raise ValueError("This approval is only for scene keyframes")
        project_id = asset["project_id"]
        source, manifest = self._visual_records(project_id)
        index = next(index for index, scene in enumerate(source.scenes) if scene.id == asset["scene_id"])
        ledger, current = self._ledger(project_id)
        approved_scene_ids = {item.scene_id for item in ledger.scenes}
        if any(scene.id not in approved_scene_ids for scene in source.scenes[:index]):
            raise ValueError("Approve every preceding scene first")
        spec = next(item for item in manifest.scenes if item.scene_id == asset["scene_id"])
        existing_set = next((item for item in ledger.sets if item.set_id == spec.set_id), None)
        if existing_set and existing_set.source_scene_id != asset["scene_id"]:
            compilations = self.repository.latest_revision(project_id, "scene_prompt_compilations")
            compilation = next((
                item for item in reversed(compilations["payload"]) if item["asset_id"] == asset_id
            ), None) if compilations else None
            referenced_ids = {
                item["asset_id"] for item in compilation.get("reference_assets", [])
            } if compilation else set()
            if existing_set.reference_asset_id not in referenced_ids and not request.override_reference_gate:
                raise ValueError(
                    "This recurring set has an approved reference. Recompile it, or explicitly confirm and document that "
                    "the existing image already matches the approved set."
                )
            if existing_set.reference_asset_id not in referenced_ids:
                reason = (request.reference_override_reason or "").strip()
                if len(reason) < 10:
                    raise ValueError("Record why this image can safely bypass same-set reference recompilation")
                history = self.repository.latest_revision(project_id, "visual_continuity_overrides")
                records = list(history["payload"]) if history else []
                records.append({
                    "asset_id": asset_id, "scene_id": asset["scene_id"], "set_id": spec.set_id,
                    "approved_set_reference_asset_id": existing_set.reference_asset_id,
                    "reason": reason, "created_at": utc_now(),
                })
                self.repository.create_revision(
                    project_id, "visual_continuity_overrides", records,
                    history["id"] if history else None,
                )
                self.repository.approve(project_id, "scene_reference_override", None, reason)
        approved_asset = self._approve_verified_asset(
            asset_id, allow_override=request.override_verification,
            override_reason=request.override_reason,
        )
        if not existing_set or request.promote_as_set_reference:
            snapshot = SetContinuitySnapshot(
                set_id=spec.set_id or "", location_id=spec.location_id, name=spec.sub_location or "Unknown set",
                reference_asset_id=asset_id, source_scene_id=asset["scene_id"],
                geometry_notes=request.geometry_notes.strip(), fixed_elements=request.fixed_elements,
                materials=request.materials, palette=request.palette, adjacency_notes=request.adjacency_notes,
                mutable_elements=request.mutable_elements, approved_at=utc_now(),
            )
            ledger.sets = [item for item in ledger.sets if item.set_id != spec.set_id] + [snapshot]
        appearance_by_id = {item.id: item for item in manifest.appearances}
        scene_snapshot = SceneContinuitySnapshot(
            scene_id=asset["scene_id"], scene_number=source.scenes[index].number,
            set_id=spec.set_id or "", asset_id=asset_id, appearance_ids=spec.appearance_ids,
            costume_ids=[appearance_by_id[item].costume_id for item in spec.appearance_ids if appearance_by_id[item].costume_id],
            prop_ids=spec.prop_ids, character_state_notes=request.character_state_notes,
            prop_state_notes=request.prop_state_notes, approved_at=utc_now(),
        )
        ledger.scenes = [item for item in ledger.scenes if item.scene_id != asset["scene_id"]] + [scene_snapshot]
        ledger.scenes.sort(key=lambda item: item.scene_number)
        self.repository.create_revision(
            project_id, "visual_continuity", ledger, current["id"] if current else None,
        )
        if len(ledger.scenes) == len(source.scenes):
            self.repository.set_stage(project_id, "visuals_ready", status="ready")
        else:
            next_scene = source.scenes[index + 1]
            active_next = next((
                item for item in reversed(self.repository.list_assets(project_id))
                if item["kind"] == "scene_keyframe" and item["scene_id"] == next_scene.id
                and item["status"] != AssetStatus.invalidated
            ), None)
            if not active_next or active_next["status"] in {AssetStatus.pending, AssetStatus.failed}:
                self.compile_scene_asset(project_id, next_scene.id)
            else:
                self.repository.set_stage(project_id, "scene_images_review", status="active")
        return approved_asset

    def _approve_verified_asset(
        self, asset_id: str, *, allow_override: bool = False, override_reason: str | None = None,
    ) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        allowed_statuses = {AssetStatus.generated, AssetStatus.failed} if allow_override else {AssetStatus.generated}
        if asset["status"] not in allowed_statuses or not asset.get("path") or not Path(asset["path"]).exists():
            raise ValueError("Only an available generated image can be approved")
        verification = self.repository.latest_revision(asset["project_id"], "visual_verification")
        match = next(
            (item for item in reversed(verification["payload"]) if item["asset_id"] == asset_id), None,
        ) if verification else None
        failed_verification = (
            not match or not match["passed"]
            or any(item["severity"] == "blocking" for item in match["issues"])
        )
        if failed_verification and not allow_override:
            raise ValueError("The asset has not passed visual verification")
        if failed_verification:
            reason = (override_reason or "").strip()
            if len(reason) < 10:
                raise ValueError("Record a specific reason before overriding visual verification")
            history = self.repository.latest_revision(asset["project_id"], "visual_verification_overrides")
            records = list(history["payload"]) if history else []
            records.append({
                "asset_id": asset_id, "reason": reason, "verification": match,
                "created_at": utc_now(),
            })
            self.repository.create_revision(
                asset["project_id"], "visual_verification_overrides", records,
                history["id"] if history else None,
            )
            self.repository.approve(asset["project_id"], "scene_visual_override", None, reason)
        return self.repository.update_asset(asset_id, AssetStatus.approved)

    def revise_asset_prompt(
        self, asset_id: str, prompt: str, expected_dependency_hash: str,
    ) -> dict[str, Any]:
        """Create a pending prompt version while preserving the exact reference bundle."""
        parent = self.repository.get_asset(asset_id)
        if parent["kind"] != "scene_keyframe":
            raise ValueError("Direct prompt revision is available only for scene keyframes")
        if self.repository.get_project(parent["project_id"])["stage"] != "scene_images_review":
            raise ValueError("Scene prompts can be revised only during sequential visual review")
        if parent["status"] == AssetStatus.invalidated:
            raise ValueError("Refresh before editing; this asset version is no longer active")
        if parent["status"] == AssetStatus.approved:
            raise ValueError("Correct an approved scene through the image-correction flow so downstream continuity is revoked safely")
        if parent["dependency_hash"] != expected_dependency_hash:
            raise ValueError("The asset changed after it was loaded; refresh before saving the prompt")
        cleaned = prompt.strip()
        if cleaned == parent["prompt"].strip():
            return parent
        dependency_hash = content_hash({
            "parent_asset_id": parent["id"], "parent_dependency_hash": parent["dependency_hash"],
            "direct_prompt_revision": cleaned,
        })
        revised = self.repository.upsert_asset(
            parent["project_id"], parent["kind"], parent["canonical_id"], cleaned,
            dependency_hash, scene_id=parent["scene_id"],
        )
        compilations = self.repository.latest_revision(parent["project_id"], "scene_prompt_compilations")
        parent_compilation = next((
            item for item in reversed(compilations["payload"]) if item["asset_id"] == parent["id"]
        ), None) if compilations else None
        if parent_compilation:
            records = list(compilations["payload"])
            records.append({
                **parent_compilation, "asset_id": revised["id"],
                "dependency_hash": dependency_hash, "created_at": utc_now(),
            })
            self.repository.create_revision(
                parent["project_id"], "scene_prompt_compilations", records, compilations["id"],
            )
        history = self.repository.latest_revision(parent["project_id"], "asset_prompt_edits")
        records = list(history["payload"]) if history else []
        records.append({
            "asset_id": revised["id"], "parent_asset_id": parent["id"],
            "before_hash": content_hash(parent["prompt"]), "after_hash": content_hash(cleaned),
            "created_at": utc_now(),
        })
        self.repository.create_revision(
            parent["project_id"], "asset_prompt_edits", records, history["id"] if history else None,
        )
        return revised

    def restore_asset_version(self, asset_id: str) -> dict[str, Any]:
        """Restore a retained scene image without calling the image provider again."""
        asset = self.repository.get_asset(asset_id)
        if asset["kind"] != "scene_keyframe" or asset["status"] != AssetStatus.invalidated:
            raise ValueError("Only a retained scene-keyframe version can be restored")
        if self.repository.get_project(asset["project_id"])["stage"] not in {"scene_images_review", "visuals_ready"}:
            raise ValueError("Scene versions can be restored only during visual review")
        if not asset["path"] or not Path(asset["path"]).exists():
            raise ValueError("This retained version has no recoverable image file")
        if not self.repository.latest_revision(asset["project_id"], "visual_continuity"):
            self.repository.create_revision(
                asset["project_id"], "visual_continuity", VisualContinuityLedger(),
            )
        verification = self.repository.latest_revision(asset["project_id"], "visual_verification")
        match = next((
            item for item in reversed(verification["payload"]) if item["asset_id"] == asset_id
        ), None) if verification else None
        restored_status = (
            AssetStatus.generated if match and match.get("passed")
            and not any(item.get("severity") == "blocking" for item in match.get("issues", []))
            else AssetStatus.failed
        )
        restored = self.repository.reactivate_asset(asset_id, restored_status)
        history = self.repository.latest_revision(asset["project_id"], "asset_restorations")
        records = list(history["payload"]) if history else []
        records.append({
            "asset_id": asset_id, "restored_status": str(restored_status), "created_at": utc_now(),
        })
        self.repository.create_revision(
            asset["project_id"], "asset_restorations", records, history["id"] if history else None,
        )
        self.repository.set_stage(asset["project_id"], "scene_images_review", status="active")
        return restored

    def generate_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        if asset["status"] == AssetStatus.approved:
            return asset
        if asset["status"] == AssetStatus.invalidated:
            raise ValueError("This asset version is invalidated; generate or approve its latest corrected version")
        project_dir = settings.asset_dir / asset["project_id"]
        project_dir.mkdir(parents=True, exist_ok=True)
        output = project_dir / f"{asset_id}.png"
        temporary = project_dir / f".{asset_id}.{uuid.uuid4().hex}.tmp.png"
        references: list[tuple[str, Path]] = []
        referenced_paths: set[Path] = set()

        def add_reference(label: str, reference_asset_id: str) -> None:
            try:
                reference = self.repository.get_asset(reference_asset_id)
            except KeyError:
                return
            if reference["project_id"] != asset["project_id"] or not reference["path"]:
                return
            path = Path(reference["path"])
            if path.exists() and path not in referenced_paths:
                references.append((label, path))
                referenced_paths.add(path)

        if asset["kind"] == "scene_keyframe":
            project = self.repository.get_project(asset["project_id"])
            ledger, _ = self._ledger(asset["project_id"])
            if project["stage"] == "scene_images_review" and ledger.scenes:
                source, _ = self._visual_records(asset["project_id"])
                index = next(index for index, scene in enumerate(source.scenes) if scene.id == asset["scene_id"])
                approved_scene_ids = {item.scene_id for item in ledger.scenes}
                if any(scene.id not in approved_scene_ids for scene in source.scenes[:index]):
                    raise ValueError("This scene is locked until every preceding scene is approved")
        corrections = self.repository.latest_revision(asset["project_id"], "asset_corrections")
        correction = next(
            (item for item in reversed(corrections["payload"]) if item["asset_id"] == asset_id), None,
        ) if corrections else None
        if correction:
            add_reference("REJECTED PREVIOUS VERSION — preserve unaffected details and correct only the requested defect", correction["parent_asset_id"])
            if correction.get("style_reference_asset_id"):
                add_reference("PROJECT STYLE REFERENCE ONLY — do not copy identity, costume or set geometry", correction["style_reference_asset_id"])
        if asset["kind"] == "scene_keyframe":
            compilations = self.repository.latest_revision(asset["project_id"], "scene_prompt_compilations")
            compilation = next((
                item for item in reversed(compilations["payload"]) if item["asset_id"] == asset_id
            ), None) if compilations else None
            if compilation:
                for row in compilation["reference_assets"]:
                    add_reference(row["label"], row["asset_id"])
            manifest_revision = self.repository.latest_revision(asset["project_id"], "visual_manifest")
            if manifest_revision:
                manifest = VisualManifest.model_validate(manifest_revision["payload"])
                spec = next((item for item in manifest.scenes if item.id == asset["canonical_id"]), None)
                if spec:
                    all_assets = self.repository.list_assets(asset["project_id"])
                    for appearance_id in spec.appearance_ids:
                        reference = next((item for item in all_assets if item["kind"] == "character_sheet" and item["canonical_id"] == appearance_id and item["status"] == AssetStatus.approved), None)
                        if reference and reference["path"]:
                            add_reference(f"IMMUTABLE CHARACTER/COSTUME REFERENCE — {appearance_id}", reference["id"])
        try:
            self.repository.update_asset(asset_id, AssetStatus.pending, increment_attempt=True)
            self.ai.provider.generate_image(asset["prompt"], temporary, references)
            os.replace(temporary, output)
            verification = self.ai.cached_visual_verification(
                asset["project_id"], asset_id, asset["dependency_hash"], asset["prompt"], output, references,
            )
            history = self.repository.latest_revision(asset["project_id"], "visual_verification")
            records = list(history["payload"]) if history else []
            records.append(verification.model_dump(mode="json"))
            self.repository.create_revision(
                asset["project_id"], "visual_verification", records, history["id"] if history else None,
            )
            if not verification.passed or any(issue.severity == "blocking" for issue in verification.issues):
                return self.repository.update_asset(
                    asset_id, AssetStatus.failed, path=str(output),
                    error="Visual verification failed: " + verification.summary,
                )
            return self.repository.update_asset(asset_id, AssetStatus.generated, path=str(output), error=None)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            # Image generation may have succeeded before the independent verifier
            # returned an empty/malformed structured response. Preserve that image
            # so the human reviewer can retry verification or approve with reason.
            preserved_path = str(output) if output.exists() else None
            self.repository.update_asset(
                asset_id, AssetStatus.failed, path=preserved_path, error=str(error),
            )
            raise

    def correct_asset(self, asset_id: str, request: AssetCorrectionRequest) -> dict[str, Any]:
        parent = self.repository.get_asset(asset_id)
        project = self.repository.get_project(parent["project_id"])
        allowed_stages = {"character_images_review"} if parent["kind"] == "character_sheet" else {"visuals_ready", "scene_images_review"}
        if project["stage"] not in allowed_stages:
            raise ValueError(f"{parent['kind']} corrections are unavailable at stage {project['stage']}")
        if parent["status"] not in {AssetStatus.generated, AssetStatus.failed, AssetStatus.approved}:
            raise ValueError("Only a generated, failed or approved asset can be corrected")
        if not parent["path"] or not Path(parent["path"]).exists():
            raise ValueError("Generate the original asset before applying a visual correction")
        style_reference = None
        if request.style_reference_asset_id:
            style_reference = self.repository.get_asset(request.style_reference_asset_id)
            if style_reference["project_id"] != parent["project_id"]:
                raise ValueError("Style reference must belong to the same project")
            if style_reference["id"] == parent["id"]:
                raise ValueError("Choose a different asset as the style reference")
            if style_reference["kind"] != parent["kind"] or style_reference["status"] not in {
                AssetStatus.generated, AssetStatus.approved,
            }:
                raise ValueError("Style reference must be another generated asset of the same kind")
            if not style_reference["path"] or not Path(style_reference["path"]).exists():
                raise ValueError("Style-reference image is unavailable")
        correction_prompt = (
            f"{VISUAL_STYLE_LOCK}\n\n"
            f"APPROVED BASE SPECIFICATION (preserve every detail not explicitly changed):\n{parent['prompt']}\n\n"
            "USER-DIRECTED SURGICAL VISUAL CORRECTION (highest priority):\n"
            f"{request.instruction.strip()}\n\n"
            "The first supplied image is the rejected previous version: preserve its useful identity/composition anchors but "
            "correct the stated defect. If a second reference is supplied, use it only for medium, lighting and project-wide "
            "visual style; never copy that other character's face, body, clothing or identity. Return the complete corrected image."
        )
        dependency_hash = content_hash({
            "parent_asset_id": parent["id"], "parent_dependency_hash": parent["dependency_hash"],
            "instruction": request.instruction.strip(),
            "style_reference_asset_id": style_reference["id"] if style_reference else None,
            "prompt": correction_prompt,
        })
        corrected = self.repository.upsert_asset(
            parent["project_id"], parent["kind"], parent["canonical_id"], correction_prompt,
            dependency_hash, scene_id=parent["scene_id"],
        )
        history = self.repository.latest_revision(parent["project_id"], "asset_corrections")
        records = list(history["payload"]) if history else []
        records.append({
            "asset_id": corrected["id"], "parent_asset_id": parent["id"],
            "style_reference_asset_id": style_reference["id"] if style_reference else None,
            "instruction": request.instruction.strip(), "created_at": utc_now(),
        })
        self.repository.create_revision(
            parent["project_id"], "asset_corrections", records, history["id"] if history else None,
        )
        if parent["kind"] == "scene_keyframe":
            compilation_history = self.repository.latest_revision(parent["project_id"], "scene_prompt_compilations")
            parent_compilation = next((
                item for item in reversed(compilation_history["payload"]) if item["asset_id"] == parent["id"]
            ), None) if compilation_history else None
            if parent_compilation:
                compilation_records = list(compilation_history["payload"])
                compilation_records.append({
                    **parent_compilation, "asset_id": corrected["id"],
                    "dependency_hash": dependency_hash, "created_at": utc_now(),
                })
                self.repository.create_revision(
                    parent["project_id"], "scene_prompt_compilations", compilation_records,
                    compilation_history["id"],
                )
        if parent["kind"] == "character_sheet":
            manifest_revision = self.repository.latest_revision(parent["project_id"], "visual_manifest")
            if manifest_revision:
                manifest = VisualManifest.model_validate(manifest_revision["payload"])
                affected_scene_assets = [
                    scene.id for scene in manifest.scenes if parent["canonical_id"] in scene.appearance_ids
                ]
                self.repository.invalidate_assets(parent["project_id"], affected_scene_assets)
        elif parent["scene_id"]:
            # Changing an approved scene revokes that scene and all later visual
            # snapshots, but never mutates an earlier set such as Scene 1's room.
            source, manifest = self._visual_records(parent["project_id"])
            target_index = next(index for index, scene in enumerate(source.scenes) if scene.id == parent["scene_id"])
            ledger, current = self._ledger(parent["project_id"])
            removed = [item for item in ledger.scenes if item.scene_number >= source.scenes[target_index].number]
            removed_asset_ids = {item.asset_id for item in removed}
            ledger.scenes = [item for item in ledger.scenes if item.asset_id not in removed_asset_ids]
            ledger.sets = [item for item in ledger.sets if item.reference_asset_id not in removed_asset_ids]
            if current:
                if removed:
                    self.repository.create_revision(parent["project_id"], "visual_continuity", ledger, current["id"])
                downstream_ids = [
                    item.id for item in manifest.scenes
                    if next(scene.number for scene in source.scenes if scene.id == item.scene_id) > source.scenes[target_index].number
                ]
                self.repository.invalidate_assets(parent["project_id"], downstream_ids)
                self.repository.set_stage(parent["project_id"], "scene_images_review", status="active")
        return self.generate_asset(corrected["id"])

    def approve_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        if asset["kind"] == "scene_keyframe":
            raise ValueError("Scene images require continuity approval and an editable snapshot")
        return self._approve_verified_asset(asset_id)


class ExportService:
    def __init__(self, repository: Repository):
        self.repository = repository

    def build(self, project_id: str) -> Path:
        project = self.repository.get_project(project_id)
        assets = self.repository.list_assets(project_id)
        active_assets = [asset for asset in assets if asset["status"] != AssetStatus.invalidated]
        if project["stage"] != "visuals_ready":
            raise ValueError("Export is available only after the visual workflow is ready")
        incomplete = [asset["id"] for asset in active_assets if asset["status"] not in {AssetStatus.generated, AssetStatus.approved}]
        if not active_assets or incomplete:
            raise ValueError(f"Generate and verify all visual assets before export; incomplete assets: {incomplete}")
        revisions = self.repository.list_latest_revisions(project_id)
        adapted = AdaptedScreenplay.model_validate(revisions["adapted_screenplay"]["payload"])
        extraction = SourceScreenplay.model_validate(revisions["extraction"]["payload"])
        character_names = {character.id: character.name for character in extraction.characters}
        export_root = settings.export_dir / project_id
        if export_root.exists():
            shutil.rmtree(export_root)
        export_root.mkdir(parents=True)
        screenplay_text = self._screenplay_text(adapted, character_names)
        (export_root / "adapted_screenplay.txt").write_text(screenplay_text, encoding="utf-8")
        screenplay_body = screenplay_text.removeprefix(f"{adapted.title}\n\n")
        self._write_pdf(export_root / "adapted_screenplay.pdf", adapted.title, screenplay_body)
        mapping = {
            "scene_breakdown.json": "extraction",
            "cultural_brief.json": "cultural_brief",
            "adaptation_plan.json": "adaptation_plan",
            "canonical_characters.json": "extraction",
            "production_bible.json": "extraction",
            "visual_manifest.json": "visual_manifest",
            "visual_continuity.json": "visual_continuity",
            "visual_continuity_overrides.json": "visual_continuity_overrides",
            "scene_prompt_compilations.json": "scene_prompt_compilations",
            "visual_verification.json": "visual_verification",
            "visual_verification_overrides.json": "visual_verification_overrides",
            "change_ledger.json": "change_ledger",
            "asset_corrections.json": "asset_corrections",
            "asset_prompt_edits.json": "asset_prompt_edits",
            "asset_restorations.json": "asset_restorations",
        }
        for filename, kind in mapping.items():
            payload = revisions[kind]["payload"] if kind in revisions else (
                [] if kind in {
                    "change_ledger", "visual_verification", "visual_verification_overrides", "visual_continuity_overrides",
                    "asset_corrections", "asset_prompt_edits", "asset_restorations", "scene_prompt_compilations",
                } else {}
            )
            if filename == "canonical_characters.json" and payload:
                payload = payload["characters"]
            if filename == "production_bible.json" and payload:
                payload = payload["production_elements"]
            (export_root / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        issues = revisions.get("continuity_issues", {"payload": []})["payload"]
        continuity_text = "\n\n".join(f"[{item.get('severity','info').upper()}] {item.get('message','')}\nAffected: {', '.join(item.get('affected_scene_ids', []))}" for item in issues) or "No continuity issues recorded."
        self._write_pdf(export_root / "continuity_report.pdf", "Continuity Report", continuity_text)
        (export_root / "ai_usage_log.json").write_text(json.dumps(self.repository.list_model_runs(project_id), ensure_ascii=False, indent=2), encoding="utf-8")
        (export_root / "known_limitations.md").write_text(
            "# Known limitations\n\n- Single-user local MVP.\n- TXT/pasted input only.\n- Maidani Mewari and Devanagari only.\n- Native-speaker validation is unavailable; uncertainty remains visible.\n- SQLite is for demonstration, not concurrent production.\n",
            encoding="utf-8",
        )
        for folder in ("character_bible", "costume_bible", "scene_keyframes"):
            (export_root / folder).mkdir()
        for asset in active_assets:
            if asset["status"] not in {AssetStatus.generated, AssetStatus.approved} or not asset["path"] or not Path(asset["path"]).exists():
                continue
            if asset["kind"] == "scene_keyframe":
                destinations = [export_root / "scene_keyframes"]
            else:
                destinations = [export_root / "character_bible", export_root / "costume_bible"]
            for destination in destinations:
                shutil.copy2(asset["path"], destination / f"{asset['canonical_id']}.png")
        zip_path = settings.export_dir / f"{project_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in export_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(export_root))
        return zip_path

    @staticmethod
    def _screenplay_text(screenplay: AdaptedScreenplay, character_names: dict[str, str] | None = None) -> str:
        character_names = character_names or {}
        lines = [screenplay.title, ""]
        for scene in screenplay.scenes:
            lines.extend([scene.heading, ""])
            for block in scene.blocks:
                if block.speaker_id:
                    lines.append(character_names.get(block.speaker_id, f"[{block.speaker_id[:8]}]").upper())
                lines.extend([block.adapted_text, ""])
        return "\n".join(lines)

    @staticmethod
    def _write_pdf(path: Path, title: str, body: str) -> None:
        from weasyprint import HTML

        font = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansDevanagari.ttf"
        if not font.exists():
            raise FileNotFoundError(f"Bundled Devanagari font is missing: {font}")
        paragraphs = "".join(f"<p>{escape(part)}</p>" for part in body.split("\n\n"))
        html = f"""<!doctype html><html lang="hi"><meta charset="utf-8"><style>
        @page {{ size: A4; margin: 22mm; }}
        @font-face {{ font-family: 'Bundled Noto Sans Devanagari'; src: url('{font.as_uri()}'); }}
        body {{ font-family: 'Bundled Noto Sans Devanagari', sans-serif; font-size: 11pt; line-height: 1.5; }}
        h1 {{ font-size: 20pt; }} p {{ white-space: pre-wrap; }}
        </style><body><h1>{escape(title)}</h1>{paragraphs}</body></html>"""
        HTML(string=html).write_pdf(path)
