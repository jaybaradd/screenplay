from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import settings
from backend.cultures.models import TargetSelection
from backend.cultures.registry import culture_registry
from backend.observability import observer
from backend.schemas import (
    ApprovalRequest, AssetCorrectionRequest, AssetPromptRevisionRequest, AssetStatus, CorrectionRequest,
    MergeRequest, ProjectCreate, RecordPatch, SceneVisualApprovalRequest,
)
from backend.services import ExportService, ProjectService
from backend.storage import Repository
from backend.workflow import Workflow


settings.ensure_directories()
repository = Repository(settings.app_db)
workflow = Workflow(repository)
project_service = ProjectService(repository, workflow.ai)
export_service = ExportService(repository)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    observer.flush()


app = FastAPI(title="Cultural Screenplay Adaptation Studio", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def view(project_id: str) -> dict[str, Any]:
    project = repository.get_project(project_id)
    project["revisions"] = repository.list_latest_revisions(project_id)
    project["approvals"] = repository.list_approvals(project_id)
    cultural_brief = project["revisions"].get("cultural_brief")
    project["approved_brief_revision_id"] = (
        cultural_brief["id"]
        if cultural_brief and any(item["gate"] == "plan" for item in project["approvals"])
        else None
    )
    assets = repository.list_assets(project_id)
    # Recover image files that were atomically written before a verifier/parser
    # failure prevented the database path from being committed.
    for asset in assets:
        if asset["status"] == AssetStatus.failed and not asset["path"]:
            candidate = settings.asset_dir / project_id / f"{asset['id']}.png"
            if candidate.exists():
                repository.update_asset(
                    asset["id"], AssetStatus.failed, path=str(candidate), error=asset["error"],
                )
    project["assets"] = repository.list_assets(project_id)
    return project


def _project_observability(project_id: str | None) -> tuple[dict[str, Any], list[str]]:
    if not project_id:
        return {}, [f"environment:{settings.langfuse_environment}"]
    try:
        project = repository.get_project(project_id)
    except KeyError:
        return {"project_id": project_id}, [f"environment:{settings.langfuse_environment}"]
    metadata = {
        "project_id": project_id, "project_title": project["title"], "stage_before": project["stage"],
        "culture_id": project["culture_id"], "culture_display_name": project["culture_display_name"],
        "profile_hash": project["profile_hash"],
    }
    tags = [
        f"environment:{settings.langfuse_environment}", f"culture:{project['culture_id']}",
        f"provider:{workflow.ai.provider.name}",
    ]
    return metadata, tags


def _asset_project(asset_id: str) -> str | None:
    try:
        return repository.get_asset(asset_id)["project_id"]
    except KeyError:
        return None


def guard(
    action, *, command_name: str | None = None, project_id: str | None = None,
    input_data: Any = None, metadata: dict[str, Any] | None = None,
):
    try:
        if not command_name:
            return action()
        project_metadata, tags = _project_observability(project_id)
        with observer.command(
            command_name, project_id=project_id, input_data=input_data,
            metadata={**project_metadata, **(metadata or {})}, tags=tags,
        ):
            result = action()
            stage_after = None
            if project_id:
                try:
                    stage_after = repository.get_project(project_id)["stage"]
                except KeyError:
                    pass
            observer.set_command_result(result="success", stage_after=stage_after)
            return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok", "ai_mode": settings.ai_mode, "provider": workflow.ai.provider.name,
        "text_model": workflow.ai.provider.text_model, "image_model": workflow.ai.provider.image_model,
        "observability": observer.health(),
    }


@app.get("/v1/projects")
def list_projects() -> list[dict[str, Any]]:
    return repository.list_projects()


@app.get("/v1/cultures")
def list_cultures() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in culture_registry.summaries()]


@app.get("/v1/cultures/{culture_id}")
def get_culture(culture_id: str) -> dict[str, Any]:
    def action():
        profile = culture_registry.snapshot(culture_id).profile
        if not profile.production_enabled:
            raise KeyError(f"Culture profile {culture_id} is not production-enabled")
        return profile.model_dump(mode="json")
    return guard(action)


@app.post("/v1/projects", status_code=201)
def create_project(request: ProjectCreate, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = "create-project"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return view(cached["id"])
    def action():
        selection = TargetSelection(
            culture_id=request.culture_id, locality=request.locality, setting=request.setting,
            period=request.period, output_script=request.output_script,
        )
        snapshot = culture_registry.validate_selection(selection)
        project = repository.create_project(request, snapshot)
        repository.save_idempotent(scope, idempotency_key, {"id": project["id"]})
        return view(project["id"])
    return guard(
        action, command_name="project.create", input_data=request.model_dump(mode="json"),
        metadata={"culture_id": request.culture_id},
    )


@app.get("/v1/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return guard(lambda: view(project_id))


@app.post("/v1/projects/{project_id}/advance")
def advance(project_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"advance:{project_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return view(project_id)

    def action():
        project = repository.get_project(project_id)
        retryable = {
            "extracting", "researching_culture", "adapting", "building_visual_manifest",
            "preparing_character_assets", "preparing_scene_assets",
        }
        if project["stage"] == "created":
            workflow.start(project_id)
        elif project["stage"] in retryable:
            workflow.retry_failed(project_id)
        else:
            raise ValueError(f"Project is at {project['stage']}; use the matching approval action")
        repository.save_idempotent(scope, idempotency_key, {"advanced": True})
        return view(project_id)
    return guard(action, command_name="workflow.advance", project_id=project_id)


@app.patch("/v1/projects/{project_id}/records/{record_kind}/{record_id}")
def patch_record(project_id: str, record_kind: str, record_id: str, request: RecordPatch, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"patch:{project_id}:{record_kind}:{record_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_revision(cached["revision_id"])
    def action():
        result = project_service.patch_record(project_id, request)
        repository.save_idempotent(scope, idempotency_key, {"revision_id": result["id"]})
        return result
    return guard(
        action, command_name="record.patch", project_id=project_id,
        input_data={"record_kind": record_kind, "record_id": record_id, "patch": request.model_dump(mode="json")},
    )


@app.post("/v1/projects/{project_id}/merges")
def merge_records(project_id: str, request: MergeRequest, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"merge:{project_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_revision(cached["revision_id"])
    def action():
        result = project_service.merge_records(project_id, request)
        repository.save_idempotent(scope, idempotency_key, {"revision_id": result["id"]})
        return result
    return guard(
        action, command_name="record.merge", project_id=project_id,
        input_data=request.model_dump(mode="json"),
    )


@app.post("/v1/projects/{project_id}/continuity/repair")
def repair_continuity(project_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"repair-continuity:{project_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_revision(cached["revision_id"])

    def action():
        revision = project_service.repair_continuity(project_id)
        repository.save_idempotent(scope, idempotency_key, {"revision_id": revision["id"]})
        return revision

    return guard(action, command_name="continuity.repair", project_id=project_id)


@app.post("/v1/projects/{project_id}/approve/{gate}")
def approve(project_id: str, gate: str, request: ApprovalRequest, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"approve:{project_id}:{gate}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return view(project_id)

    def action():
        project = repository.get_project(project_id)
        expected_stage = {
            "extraction": "extraction_review", "plan": "plan_review", "screenplay": "screenplay_review",
            "visuals": "visual_review", "character_images": "character_images_review",
        }.get(gate)
        if not expected_stage or project["stage"] != expected_stage:
            raise ValueError(f"Gate {gate} is not active at stage {project['stage']}")
        if gate == "extraction":
            issues = repository.latest_revision(project_id, "continuity_issues")
            blocking = [item for item in (issues["payload"] if issues else []) if item.get("severity") == "blocking"]
            if blocking and not request.override_reason:
                raise ValueError("Resolve blocking continuity issues or record an override reason")
        if gate == "screenplay":
            verification = repository.latest_revision(project_id, "adaptation_verification")
            blocking = [item for item in (verification["payload"] if verification else []) if item.get("severity") == "blocking"]
            if blocking and not request.override_reason:
                raise ValueError("Resolve blocking adaptation issues or record an override reason")
        if gate == "plan":
            verification = repository.latest_revision(project_id, "plan_verification")
            blocking = [item for item in (verification["payload"] if verification else []) if item.get("severity") == "blocking"]
            if blocking and not request.override_reason:
                raise ValueError("Resolve blocking cultural/verbal-plan issues or record an override reason")
        if gate == "character_images":
            sheets = [
                asset for asset in repository.list_assets(project_id)
                if asset["kind"] == "character_sheet" and asset["status"] != AssetStatus.invalidated
            ]
            if not sheets or any(asset["status"] != AssetStatus.approved for asset in sheets):
                raise ValueError("Generate and approve every character sheet first")
        revision_id = request.revision_id
        if not revision_id:
            kind = {"extraction": "extraction", "plan": "adaptation_plan", "screenplay": "adapted_screenplay", "visuals": "visual_manifest"}.get(gate)
            current = repository.latest_revision(project_id, kind) if kind else None
            revision_id = current["id"] if current else None
        if gate == "visuals":
            if not revision_id:
                raise ValueError("A visual manifest revision is required")
            workflow.validate_visual_manifest_revision(project_id, revision_id)
        repository.approve(project_id, gate, revision_id, request.override_reason)
        observer.score(
            "human_approval", 1.0, project_id=project_id,
            comment=f"gate={gate}; override={bool(request.override_reason)}",
        )
        workflow.resume(project_id, {"action": "approve", "revision_id": revision_id, "override_reason": request.override_reason})
        repository.save_idempotent(scope, idempotency_key, {"approved": True})
        return view(project_id)
    return guard(
        action, command_name="workflow.approve", project_id=project_id,
        input_data=request.model_dump(mode="json"), metadata={"gate": gate},
    )


@app.post("/v1/projects/{project_id}/corrections")
def correct(project_id: str, request: CorrectionRequest, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"correct:{project_id}:{request.target_block_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_revision(cached["revision_id"])
    def action():
        project = repository.get_project(project_id)
        if project["stage"] != "screenplay_review":
            raise ValueError("Surgical corrections are available only during screenplay review")
        result = project_service.correct_block(project_id, request)
        repository.save_idempotent(scope, idempotency_key, {"revision_id": result["id"]})
        return result
    return guard(
        action, command_name="screenplay.correct", project_id=project_id,
        input_data=request.model_dump(mode="json"), metadata={"target_block_id": request.target_block_id},
    )


@app.post("/v1/projects/{project_id}/visuals/specs")
def visual_specs(project_id: str) -> dict[str, Any]:
    return guard(lambda: repository.latest_revision(project_id, "visual_manifest") or {})


@app.post("/v1/projects/{project_id}/visuals/regenerate")
def regenerate_visual_specs(project_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"regenerate-visual-manifest:{project_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_revision(cached["revision_id"])

    def action():
        revision = workflow.regenerate_visual_manifest(project_id)
        repository.save_idempotent(scope, idempotency_key, {"revision_id": revision["id"]})
        return revision

    return guard(action, command_name="visual.manifest.regenerate", project_id=project_id)


@app.post("/v1/projects/{project_id}/visuals/continuity/start")
def start_visual_continuity(project_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"start-visual-continuity:{project_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return view(project_id)

    def action():
        project_service.start_visual_continuity(project_id)
        repository.save_idempotent(scope, idempotency_key, {"started": True})
        return view(project_id)

    return guard(action, command_name="visual.continuity.start", project_id=project_id)


@app.post("/v1/projects/{project_id}/visuals/scenes/{scene_id}/compile")
def compile_scene(project_id: str, scene_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"compile-scene:{project_id}:{scene_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(cached["asset_id"])

    def action():
        result = project_service.compile_scene_asset(project_id, scene_id)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": result["id"]})
        return result

    return guard(
        action, command_name="visual.scene.compile", project_id=project_id,
        metadata={"scene_id": scene_id},
    )


@app.post("/v1/assets/{asset_id}/generate")
def generate_asset(asset_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"generate:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)
    def action():
        result = project_service.generate_asset(asset_id)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id, "attempt": result["attempt"]})
        return result
    return guard(
        action, command_name="asset.generate", project_id=_asset_project(asset_id),
        metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/retry")
def retry_asset(asset_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"retry:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)

    def action():
        result = project_service.generate_asset(asset_id)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id, "attempt": result["attempt"]})
        return result
    return guard(
        action, command_name="asset.retry", project_id=_asset_project(asset_id),
        metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/correct")
def correct_asset(
    asset_id: str, request: AssetCorrectionRequest, idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    scope = f"correct-asset:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(cached["asset_id"])

    def action():
        result = project_service.correct_asset(asset_id, request)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": result["id"]})
        return result

    return guard(
        action, command_name="asset.correct", project_id=_asset_project(asset_id),
        input_data=request.model_dump(mode="json"), metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/revise-prompt")
def revise_asset_prompt(
    asset_id: str, request: AssetPromptRevisionRequest,
    idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    scope = f"revise-asset-prompt:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(cached["asset_id"])

    def action():
        result = project_service.revise_asset_prompt(
            asset_id, request.prompt, request.expected_dependency_hash,
        )
        repository.save_idempotent(scope, idempotency_key, {"asset_id": result["id"]})
        return result

    return guard(
        action, command_name="visual.prompt.revise", project_id=_asset_project(asset_id),
        input_data=request.model_dump(mode="json"), metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/restore")
def restore_asset_version(asset_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"restore-asset-version:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)

    def action():
        result = project_service.restore_asset_version(asset_id)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id})
        return result

    return guard(
        action, command_name="asset.restore", project_id=_asset_project(asset_id),
        metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/approve")
def approve_asset(asset_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"approve-asset:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)

    def action():
        result = project_service.approve_asset(asset_id)
        observer.score("human_asset_approval", 1.0, project_id=result["project_id"])
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id})
        return result
    return guard(
        action, command_name="asset.approve", project_id=_asset_project(asset_id),
        metadata={"asset_id": asset_id},
    )


@app.post("/v1/assets/{asset_id}/approve-scene")
def approve_scene_asset(
    asset_id: str, request: SceneVisualApprovalRequest,
    idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    scope = f"approve-scene:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)

    def action():
        result = project_service.approve_scene_asset(asset_id, request)
        observer.score(
            "human_asset_approval", 1.0, project_id=result["project_id"],
            comment=f"verification_override={request.override_verification}",
        )
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id})
        return result

    return guard(
        action, command_name=("visual.verification.override" if request.override_verification else "asset.approve"),
        project_id=_asset_project(asset_id),
        input_data=request.model_dump(mode="json"), metadata={"asset_id": asset_id, "asset_kind": "scene_keyframe"},
    )


@app.get("/v1/assets/{asset_id}/content")
def asset_content(asset_id: str):
    def action():
        asset = repository.get_asset(asset_id)
        if not asset["path"] or not Path(asset["path"]).exists():
            raise KeyError("Asset file does not exist")
        return FileResponse(asset["path"], media_type="image/png", filename=f"{asset_id}.png")
    return guard(action)


@app.get("/v1/projects/{project_id}/export")
def export_project(project_id: str):
    def action():
        path = export_service.build(project_id)
        return FileResponse(path, media_type="application/zip", filename="project-export.zip")
    return guard(action, command_name="export.build", project_id=project_id)
