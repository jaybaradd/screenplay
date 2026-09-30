from __future__ import annotations

from pathlib import Path

import backend.config as config_module
import backend.providers as providers_module
import backend.services as services_module
import backend.workflow as workflow_module
from backend.config import Settings
from backend.schemas import AdaptedBlock, AdaptedScene, BlockType, ContentBlock, CorrectionRequest, ProjectCreate, RecordPatch, SceneRecord
from backend.services import ExportService, ProjectService
from backend.storage import Repository, content_hash
from backend.workflow import Workflow


def configured_workflow(tmp_path: Path):
    local_settings = Settings(data_dir=tmp_path, ai_mode="mock", gemini_api_key=None)
    for module in (config_module, providers_module, services_module, workflow_module):
        module.settings = local_settings
    local_settings.ensure_directories()
    repository = Repository(local_settings.app_db)
    workflow = Workflow(repository)
    return local_settings, repository, workflow


def sample_text() -> str:
    return (Path(__file__).parent / "fixtures" / "sample_screenplay.txt").read_text(encoding="utf-8")


def test_lossless_scene_mapping_rejects_omitted_blocks():
    source = SceneRecord(
        id="scene-1", number=1, heading="INT. ROOM - DAY", location="ROOM",
        summary="A test scene.", dramatic_purpose="Test mapping.",
        blocks=[
            ContentBlock(id="block-1", type=BlockType.action, text="He enters."),
            ContentBlock(id="block-2", type=BlockType.dialogue, text="Wait.", speaker_id="character-1"),
        ],
    )
    incomplete = AdaptedScene(
        id="scene-1", source_scene_id="scene-1", heading="अंदर", summary="परीक्षण",
        blocks=[AdaptedBlock(
            id="block-1", source_block_ids=["block-1"], type=BlockType.action,
            adapted_text="वह अंदर आवे है।", adaptation_layer_ids=[], cultural_claim_ids=[],
            explanation="Adapted action.", confidence="high", changed_dimensions=[],
        )],
    )
    errors = Workflow._scene_mapping_errors(source, incomplete)
    assert any("expected 2 blocks, got 1" in error for error in errors)
    assert any("missing source block" in error for error in errors)


def test_complete_durable_mock_workflow_and_surgical_edit(tmp_path: Path):
    settings, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="The Letter", source_text=sample_text(), locality="Rajsamand plains", setting="rural",
    ))
    project_id = project["id"]

    workflow.start(project_id)
    assert repository.get_project(project_id)["stage"] == "extraction_review"

    extraction = repository.latest_revision(project_id, "extraction")
    assert len(extraction["payload"]["scenes"]) == 3
    workflow.resume(project_id, {"action": "approve", "revision_id": extraction["id"]})
    assert repository.get_project(project_id)["stage"] == "plan_review"
    assert len(repository.latest_revision(project_id, "adaptation_plan")["payload"]["layers"]) == 6

    plan = repository.latest_revision(project_id, "adaptation_plan")
    workflow.resume(project_id, {"action": "approve", "revision_id": plan["id"]})
    assert repository.get_project(project_id)["stage"] == "screenplay_review"

    adapted = repository.latest_revision(project_id, "adapted_screenplay")
    blocks = [block for scene in adapted["payload"]["scenes"] for block in scene["blocks"]]
    target = blocks[0]
    untouched_before = {block["id"]: content_hash(block) for block in blocks[1:]}
    project_service = ProjectService(repository, workflow.ai)
    corrected = project_service.correct_block(project_id, CorrectionRequest(
        target_block_id=target["id"], instruction="Exact corrected mock text.", expected_hash=content_hash(target),
    ))
    corrected_blocks = [block for scene in corrected["payload"]["scenes"] for block in scene["blocks"]]
    assert corrected_blocks[0]["adapted_text"] == "Exact corrected mock text."
    assert {block["id"]: content_hash(block) for block in corrected_blocks[1:]} == untouched_before

    workflow.resume(project_id, {"action": "approve", "revision_id": corrected["id"]})
    assert repository.get_project(project_id)["stage"] == "visual_review"
    visual = repository.latest_revision(project_id, "visual_manifest")
    workflow.resume(project_id, {"action": "approve", "revision_id": visual["id"]})
    assert repository.get_project(project_id)["stage"] == "character_images_review"

    character_assets = [item for item in repository.list_assets(project_id) if item["kind"] == "character_sheet"]
    assert character_assets
    for asset in character_assets:
        generated = project_service.generate_asset(asset["id"])
        assert Path(generated["path"]).exists()
        project_service.approve_asset(asset["id"])
    visual_checks = repository.latest_revision(project_id, "visual_verification")
    assert visual_checks and len(visual_checks["payload"]) == len(character_assets)
    assert all(item["passed"] for item in visual_checks["payload"])
    workflow.resume(project_id, {"action": "approve"})
    assert repository.get_project(project_id)["stage"] == "visuals_ready"

    scene_assets = [item for item in repository.list_assets(project_id) if item["kind"] == "scene_keyframe"]
    assert len(scene_assets) == 3
    for asset in scene_assets:
        project_service.generate_asset(asset["id"])
    first_attempts = {
        item["id"]: item["attempt"]
        for item in repository.list_assets(project_id) if item["kind"] == "scene_keyframe"
    }
    project_service.generate_asset(scene_assets[0]["id"])
    after = {item["id"]: item["attempt"] for item in repository.list_assets(project_id) if item["kind"] == "scene_keyframe"}
    assert after[scene_assets[0]["id"]] == first_attempts[scene_assets[0]["id"]] + 1
    assert all(after[item["id"]] == first_attempts[item["id"]] for item in scene_assets[1:])

    export = ExportService(repository).build(project_id)
    assert export.exists()


def test_checkpoint_resume_after_workflow_recreation(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Restart", source_text=sample_text(), locality="Rajsamand plains", setting="urban",
    ))
    workflow.start(project["id"])
    assert repository.get_project(project["id"])["stage"] == "extraction_review"
    workflow._checkpoint_connection.close()

    recreated = Workflow(repository)
    extraction = repository.latest_revision(project["id"], "extraction")
    recreated.resume(project["id"], {"action": "approve", "revision_id": extraction["id"]})
    assert repository.get_project(project["id"])["stage"] == "plan_review"


def test_extraction_edit_recalculates_continuity(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Continuity refresh", source_text=sample_text(), locality="Rajsamand plains", setting="rural",
    ))
    workflow.start(project["id"])
    extraction = repository.latest_revision(project["id"], "extraction")
    service = ProjectService(repository, workflow.ai)
    service.patch_record(project["id"], RecordPatch(
        document_kind="extraction", field_path="/scenes/1/number", value=9,
        expected_hash=extraction["sha256"],
    ))
    issues = repository.latest_revision(project["id"], "continuity_issues")
    assert any(item["code"] == "scene_order" and item["severity"] == "blocking" for item in issues["payload"])


def test_model_cache_is_project_scoped(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    projects = [repository.create_project(ProjectCreate(
        title=f"Project {index}", source_text="A" * 120,
        locality="Rajsamand plains", setting="rural",
    )) for index in range(2)]
    for project in projects:
        workflow.ai.cached_research(project["id"], "same-operation", "identical prompt")
    assert all(len(repository.list_model_runs(project["id"])) == 1 for project in projects)
