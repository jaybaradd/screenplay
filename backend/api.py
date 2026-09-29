from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import settings
from backend.schemas import (
    ApprovalRequest, AssetStatus, CorrectionRequest, MergeRequest, ProjectCreate, RecordPatch,
)
from backend.services import ExportService, ProjectService
from backend.storage import Repository
from backend.workflow import Workflow


settings.ensure_directories()
repository = Repository(settings.app_db)
workflow = Workflow(repository)
project_service = ProjectService(repository, workflow.ai)
export_service = ExportService(repository)

app = FastAPI(title="Maidani Mewari Screenplay Studio", version="0.1.0")
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
    project["assets"] = repository.list_assets(project_id)
    project["approvals"] = repository.list_approvals(project_id)
    return project


def guard(action):
    try:
        return action()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ai_mode": settings.ai_mode, "provider": workflow.ai.provider.name}


@app.get("/v1/projects")
def list_projects() -> list[dict[str, Any]]:
    return repository.list_projects()


@app.post("/v1/projects", status_code=201)
def create_project(request: ProjectCreate, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = "create-project"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return view(cached["id"])
    project = repository.create_project(request)
    repository.save_idempotent(scope, idempotency_key, {"id": project["id"]})
    return view(project["id"])


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
    return guard(action)


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
    return guard(action)


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
    return guard(action)


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
        if gate == "character_images":
            sheets = [asset for asset in repository.list_assets(project_id) if asset["kind"] == "character_sheet"]
            if not sheets or any(asset["status"] != AssetStatus.approved for asset in sheets):
                raise ValueError("Generate and approve every character sheet first")
        revision_id = request.revision_id
        if not revision_id:
            kind = {"extraction": "extraction", "plan": "adaptation_plan", "screenplay": "adapted_screenplay", "visuals": "visual_manifest"}.get(gate)
            current = repository.latest_revision(project_id, kind) if kind else None
            revision_id = current["id"] if current else None
        repository.approve(project_id, gate, revision_id, request.override_reason)
        workflow.resume(project_id, {"action": "approve", "revision_id": revision_id, "override_reason": request.override_reason})
        repository.save_idempotent(scope, idempotency_key, {"approved": True})
        return view(project_id)
    return guard(action)


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
    return guard(action)


@app.post("/v1/projects/{project_id}/visuals/specs")
def visual_specs(project_id: str) -> dict[str, Any]:
    return guard(lambda: repository.latest_revision(project_id, "visual_manifest") or {})


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
    return guard(action)


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
    return guard(action)


@app.post("/v1/assets/{asset_id}/approve")
def approve_asset(asset_id: str, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    scope = f"approve-asset:{asset_id}"
    cached = repository.get_idempotent(scope, idempotency_key)
    if cached:
        return repository.get_asset(asset_id)

    def action():
        result = project_service.approve_asset(asset_id)
        repository.save_idempotent(scope, idempotency_key, {"asset_id": asset_id})
        return result
    return guard(action)


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
    return guard(action)
