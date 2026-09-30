from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.config as config_module
import backend.providers as providers_module
import backend.services as services_module
import backend.workflow as workflow_module
from backend.config import Settings
from backend.schemas import (
    AdaptedBlock, AdaptedScene, AssetCorrectionRequest, AssetStatus, BlockType, ContentBlock,
    ContinuityEvent, ContinuityEventExtraction, CorrectionRequest, ProjectCreate, RecordPatch,
    SceneRecord, SceneVisualApprovalRequest,
    VisualVerification,
)
from backend.services import ExportService, ProjectService
from backend.storage import Repository, content_hash
from backend.workflow import Workflow


def test_gemini_structured_parser_handles_dict_candidate_text_and_empty_response():
    payload = {
        "asset_id": "asset-1", "dependency_hash": "hash-1", "passed": True,
        "issues": [], "summary": "Accepted",
    }
    from_dict = providers_module.parse_structured_response(
        SimpleNamespace(parsed=payload, text=None, candidates=[]), VisualVerification,
    )
    assert from_dict.passed

    candidate_response = SimpleNamespace(
        parsed=None, text=None,
        candidates=[SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[SimpleNamespace(text=__import__("json").dumps(payload))]),
        )],
    )
    assert providers_module.parse_structured_response(candidate_response, VisualVerification).asset_id == "asset-1"

    with pytest.raises(RuntimeError, match="no structured text"):
        providers_module.parse_structured_response(
            SimpleNamespace(parsed=None, text=None, candidates=[]), VisualVerification,
        )


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
        title="The Letter", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
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
    assert repository.get_project(project_id)["stage"] == "scene_images_review"

    source = repository.latest_revision(project_id, "extraction")["payload"]
    for scene in source["scenes"]:
        asset = next(
            item for item in repository.list_assets(project_id)
            if item["kind"] == "scene_keyframe" and item["scene_id"] == scene["id"]
            and item["status"] != "invalidated"
        )
        generated = project_service.generate_asset(asset["id"])
        project_service.approve_scene_asset(generated["id"], SceneVisualApprovalRequest(
            geometry_notes=f"Approved geometry for {scene['location']}",
            promote_as_set_reference=True,
        ))
    assert repository.get_project(project_id)["stage"] == "visuals_ready"
    scene_assets = [
        item for item in repository.list_assets(project_id)
        if item["kind"] == "scene_keyframe" and item["status"] == "approved"
    ]
    assert len(scene_assets) == 3

    export = ExportService(repository).build(project_id)
    assert export.exists()


def test_scoped_continuity_repair_preserves_extracted_story_records(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Continuity repair", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    ))
    workflow.start(project["id"])
    before = repository.latest_revision(project["id"], "extraction")
    service = ProjectService(repository, workflow.ai)
    repaired = service.repair_continuity(project["id"])
    assert repaired["version"] == before["version"] + 1
    for field in ("scenes", "characters", "production_elements", "story_contract", "original_text"):
        assert repaired["payload"][field] == before["payload"][field]
    assert repaired["payload"]["continuity_events"]
    assert len(repaired["payload"]["state_transitions"]) == sum(
        len(scene["character_ids"]) for scene in repaired["payload"]["scenes"]
    )
    issues = repository.latest_revision(project["id"], "continuity_issues")["payload"]
    assert not [item for item in issues if item["severity"] == "blocking"]


def test_invalid_continuity_edit_is_rejected_before_revision_is_saved(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Invalid continuity edit", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    ))
    workflow.start(project["id"])
    current = repository.latest_revision(project["id"], "extraction")
    scene = current["payload"]["scenes"][0]
    service = ProjectService(repository, workflow.ai)
    with pytest.raises(ValueError, match="Invalid continuity events"):
        service.patch_record(project["id"], RecordPatch(
            document_kind="extraction", field_path="/continuity_events", expected_hash=current["sha256"],
            value=[{
                "id": "invalid-event", "scene_id": scene["id"], "character_id": "unknown-character",
                "kind": "acquire_prop", "timing": "during_scene", "prop_id": "unknown-prop",
                "related_prop_id": None, "counterparty_character_id": None, "fact": None,
                "value": None, "relationship_character_id": None,
                "evidence_block_ids": [scene["blocks"][0]["id"]],
                "description": "Invalid references", "confidence": "low",
            }],
        ))
    assert repository.latest_revision(project["id"], "extraction")["id"] == current["id"]


def test_continuity_repair_retries_semantically_invalid_ids(tmp_path: Path, monkeypatch):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Semantic repair retry", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    ))
    workflow.start(project["id"])
    source = repository.latest_revision(project["id"], "extraction")["payload"]
    calls = {"count": 0}

    def structured(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ContinuityEventExtraction(events=[ContinuityEvent(
                id="invalid", scene_id=source["scenes"][0]["id"], character_id="unknown-character",
                kind="no_change", description="Invalid reference",
            )])
        return ContinuityEventExtraction(events=[
            ContinuityEvent(
                id=f"valid-{scene['id']}-{character_id}", scene_id=scene["id"],
                character_id=character_id, kind="no_change", description="Reviewed no durable change.",
            )
            for scene in source["scenes"] for character_id in scene["character_ids"]
        ])

    workflow.ai.provider = SimpleNamespace(name="test-live-provider")
    monkeypatch.setattr(workflow.ai, "cached_structured", structured)
    repaired = ProjectService(repository, workflow.ai).repair_continuity(project["id"])
    assert calls["count"] == 2
    assert repaired["payload"]["continuity_events"]


def test_checkpoint_resume_after_workflow_recreation(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Restart", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="urban", period="contemporary_2020_2026", output_script="devanagari",
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
        title="Continuity refresh", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
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


def test_asset_feedback_creates_one_corrected_version_without_regenerating_others(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Visual correction", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    project_id = project["id"]
    workflow.start(project_id)
    extraction = repository.latest_revision(project_id, "extraction")
    workflow.resume(project_id, {"action": "approve", "revision_id": extraction["id"]})
    plan = repository.latest_revision(project_id, "adaptation_plan")
    workflow.resume(project_id, {"action": "approve", "revision_id": plan["id"]})
    adapted = repository.latest_revision(project_id, "adapted_screenplay")
    workflow.resume(project_id, {"action": "approve", "revision_id": adapted["id"]})
    visual = repository.latest_revision(project_id, "visual_manifest")
    workflow.resume(project_id, {"action": "approve", "revision_id": visual["id"]})

    service = ProjectService(repository, workflow.ai)
    sheets = [item for item in repository.list_assets(project_id) if item["kind"] == "character_sheet"]
    first = service.generate_asset(sheets[0]["id"])
    second = service.generate_asset(sheets[1]["id"])
    corrected = service.correct_asset(first["id"], AssetCorrectionRequest(
        instruction="Use photorealistic live-action style while preserving identity and costume.",
        style_reference_asset_id=second["id"],
    ))

    assert repository.get_asset(first["id"])["status"] == "invalidated"
    assert corrected["status"] == "generated"
    assert corrected["canonical_id"] == first["canonical_id"]
    assert corrected["id"] != first["id"]
    assert repository.get_asset(second["id"])["attempt"] == second["attempt"]
    correction_log = repository.latest_revision(project_id, "asset_corrections")
    assert correction_log["payload"][-1]["asset_id"] == corrected["id"]


def test_model_cache_is_project_scoped(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    projects = [repository.create_project(ProjectCreate(
        title=f"Project {index}", source_text="A" * 120,
        culture_id="maidani_mewari", locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    )) for index in range(2)]
    for project in projects:
        workflow.ai.cached_research(project["id"], "same-operation", "identical prompt")
    assert all(len(repository.list_model_runs(project["id"])) == 1 for project in projects)


def test_exact_sub_locations_get_stable_distinct_set_ids():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = providers_module.MockProvider().extract(
        project_id, "Sets",
        "INT. FARMHOUSE - DAY\nAction.\n\nEXT. FARMHOUSE - DAY\nAction.\n\nINT. FARMHOUSE - NIGHT\nAction.",
    )
    shared_location = source.scenes[0].location_id
    for scene in source.scenes:
        scene.location_id = shared_location
        scene.location = "Farmhouse"
    source.scenes[0].sub_location = "Dining Room"
    source.scenes[1].sub_location = "Veranda"
    source.scenes[2].sub_location = "Dining Room"
    _, scaffolds = Workflow._visual_scaffolds(project_id, source)
    assert scaffolds[0]["set_id"] == scaffolds[2]["set_id"]
    assert scaffolds[0]["set_id"] != scaffolds[1]["set_id"]


def test_scene_three_compilation_uses_same_set_and_prior_scene_references(tmp_path: Path):
    _, repository, workflow = configured_workflow(tmp_path)
    project = repository.create_project(ProjectCreate(
        title="Set continuity", source_text=sample_text(), culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    project_id = project["id"]
    workflow.start(project_id)
    extraction = repository.latest_revision(project_id, "extraction")
    # Make Scenes 1 and 3 the dining room, and Scene 2 the veranda, under one parent location.
    payload = extraction["payload"]
    shared_location = payload["scenes"][0]["location_id"]
    for scene, sub_location in zip(payload["scenes"], ["Dining Room", "Veranda", "Dining Room"]):
        scene["location_id"] = shared_location
        scene["location"] = "Farmhouse"
        scene["sub_location"] = sub_location
    extraction = repository.create_revision(project_id, "extraction", payload, extraction["id"])
    workflow.resume(project_id, {"action": "approve", "revision_id": extraction["id"]})
    plan = repository.latest_revision(project_id, "adaptation_plan")
    workflow.resume(project_id, {"action": "approve", "revision_id": plan["id"]})
    adapted = repository.latest_revision(project_id, "adapted_screenplay")
    workflow.resume(project_id, {"action": "approve", "revision_id": adapted["id"]})
    visual = repository.latest_revision(project_id, "visual_manifest")
    workflow.resume(project_id, {"action": "approve", "revision_id": visual["id"]})
    service = ProjectService(repository, workflow.ai)
    for sheet in [item for item in repository.list_assets(project_id) if item["kind"] == "character_sheet"]:
        service.approve_asset(service.generate_asset(sheet["id"])["id"])
    workflow.resume(project_id, {"action": "approve"})

    scenes = payload["scenes"]
    for scene in scenes[:2]:
        scene_asset = next(
            item for item in repository.list_assets(project_id)
            if item["kind"] == "scene_keyframe" and item["scene_id"] == scene["id"]
            and item["status"] != "invalidated"
        )
        generated = service.generate_asset(scene_asset["id"])
        service.approve_scene_asset(generated["id"], SceneVisualApprovalRequest(
            geometry_notes="one east door and one north window" if scene["number"] == 1 else "open veranda",
            adjacency_notes=["Dining-room east door opens to veranda"],
            promote_as_set_reference=True,
        ))
    third = service.compile_scene_asset(project_id, scenes[2]["id"])
    compilation = repository.latest_revision(project_id, "scene_prompt_compilations")["payload"][-1]
    labels = [item["label"] for item in compilation["reference_assets"]]
    assert any("SAME-SET" in label for label in labels)
    assert any("IMMEDIATE PRIOR SCENE" in label for label in labels)
    assert "one east door and one north window" in third["prompt"]


def test_human_can_override_false_negative_visual_verification(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Override", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    image = tmp_path / "acceptable.png"
    image.write_bytes(b"acceptable-image-fixture")
    asset = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-1", "A sufficiently detailed scene prompt " * 5,
        "dependency-1", scene_id="scene-1",
    )
    repository.update_asset(asset["id"], AssetStatus.failed, path=str(image), error="Verifier rejected hand overlap")
    repository.create_revision(project["id"], "visual_verification", [{
        "asset_id": asset["id"], "dependency_hash": "dependency-1", "passed": False,
        "issues": [{
            "severity": "blocking", "dimension": "composition", "expected": "slight hand overlap",
            "observed": "overlap unclear", "message": "Minute contact could not be confirmed",
        }],
        "summary": "Hand overlap was not clear enough for the automated verifier.",
    }])
    service = ProjectService(repository, None)
    approved = service._approve_verified_asset(
        asset["id"], allow_override=True,
        override_reason="Human review confirms the subtle hand overlap is acceptable and all continuity anchors match.",
    )
    assert approved["status"] == "approved"
    overrides = repository.latest_revision(project["id"], "visual_verification_overrides")
    assert overrides["payload"][-1]["asset_id"] == asset["id"]


def test_direct_scene_prompt_edit_creates_only_one_pending_version(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Prompt edit", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    repository.set_stage(project["id"], "scene_images_review")
    first = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-1", "Original detailed visual prompt " * 5,
        "dependency-1", scene_id="scene-1",
    )
    unrelated = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-2", "Unrelated scene prompt " * 5,
        "dependency-2", scene_id="scene-2",
    )
    service = ProjectService(repository, None)
    revised = service.revise_asset_prompt(
        first["id"], "Revised prompt preserving everything except the minute hand contact. " * 4,
        first["dependency_hash"],
    )
    assert revised["status"] == "pending"
    assert repository.get_asset(first["id"])["status"] == "invalidated"
    assert repository.get_asset(unrelated["id"])["status"] == "pending"
    assert repository.latest_revision(project["id"], "asset_prompt_edits")["payload"][-1]["asset_id"] == revised["id"]


def test_retained_scene_version_can_be_restored_without_generation(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Restore", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    repository.set_stage(project["id"], "scene_images_review")
    image = tmp_path / "acceptable.png"
    image.write_bytes(b"retained-acceptable-image")
    retained = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-1", "Acceptable prompt " * 8,
        "dependency-old", scene_id="scene-1",
    )
    repository.update_asset(retained["id"], AssetStatus.generated, path=str(image))
    newer = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-1", "Later prompt " * 8,
        "dependency-new", scene_id="scene-1",
    )
    assert repository.get_asset(retained["id"])["status"] == "invalidated"
    restored = ProjectService(repository, None).restore_asset_version(retained["id"])
    assert restored["status"] == "failed"  # no passing verification: requires a human override
    assert repository.get_asset(newer["id"])["status"] == "invalidated"
    assert restored["attempt"] == 0
