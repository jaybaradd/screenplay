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
from backend.prompts import correction_prompt
from backend.providers import AIService, MockProvider
from backend.schemas import (
    AdaptedScreenplay, AssetStatus, CorrectionPatch, CorrectionRequest, MergeRequest,
    RecordPatch, SourceScreenplay, VisualManifest,
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
        current = self.repository.latest_revision(project_id, patch.document_kind)
        if not current:
            raise KeyError(f"No {patch.document_kind} document exists")
        if patch.expected_hash and patch.expected_hash != current["sha256"]:
            raise ValueError("The document changed after it was loaded; refresh before editing")
        payload = copy.deepcopy(current["payload"])
        _set_path(payload, patch.field_path, patch.value)
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

    def generate_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        if asset["status"] == AssetStatus.approved:
            return asset
        project_dir = settings.asset_dir / asset["project_id"]
        project_dir.mkdir(parents=True, exist_ok=True)
        output = project_dir / f"{asset_id}.png"
        temporary = project_dir / f".{asset_id}.{uuid.uuid4().hex}.tmp.png"
        references: list[Path] = []
        if asset["kind"] == "scene_keyframe":
            manifest_revision = self.repository.latest_revision(asset["project_id"], "visual_manifest")
            if manifest_revision:
                manifest = VisualManifest.model_validate(manifest_revision["payload"])
                spec = next((item for item in manifest.scenes if item.id == asset["canonical_id"]), None)
                if spec:
                    all_assets = self.repository.list_assets(asset["project_id"])
                    for appearance_id in spec.appearance_ids:
                        reference = next((item for item in all_assets if item["kind"] == "character_sheet" and item["canonical_id"] == appearance_id and item["status"] == AssetStatus.approved), None)
                        if reference and reference["path"]:
                            references.append(Path(reference["path"]))
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
            self.repository.update_asset(asset_id, AssetStatus.failed, error=str(error))
            raise

    def approve_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.repository.get_asset(asset_id)
        if asset["status"] != AssetStatus.generated:
            raise ValueError("Only a generated asset can be approved")
        verification = self.repository.latest_revision(asset["project_id"], "visual_verification")
        match = next(
            (item for item in reversed(verification["payload"]) if item["asset_id"] == asset_id),
            None,
        ) if verification else None
        if not match or not match["passed"] or any(item["severity"] == "blocking" for item in match["issues"]):
            raise ValueError("The asset has not passed visual verification")
        return self.repository.update_asset(asset_id, AssetStatus.approved)


class ExportService:
    def __init__(self, repository: Repository):
        self.repository = repository

    def build(self, project_id: str) -> Path:
        project = self.repository.get_project(project_id)
        assets = self.repository.list_assets(project_id)
        if project["stage"] != "visuals_ready":
            raise ValueError("Export is available only after the visual workflow is ready")
        incomplete = [asset["id"] for asset in assets if asset["status"] not in {AssetStatus.generated, AssetStatus.approved}]
        if not assets or incomplete:
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
            "visual_verification.json": "visual_verification",
            "change_ledger.json": "change_ledger",
        }
        for filename, kind in mapping.items():
            payload = revisions[kind]["payload"] if kind in revisions else ([] if kind in {"change_ledger", "visual_verification"} else {})
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
        for asset in assets:
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
