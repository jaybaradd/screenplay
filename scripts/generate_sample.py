from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import backend.config as config_module
import backend.providers as providers_module
import backend.services as services_module
import backend.workflow as workflow_module
from backend.config import Settings
from backend.schemas import ProjectCreate, SceneVisualApprovalRequest
from backend.services import ExportService, ProjectService
from backend.storage import Repository
from backend.workflow import Workflow


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_screenplay.txt"
OUTPUT = ROOT / "sample_output"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mewari-sample-") as temporary:
        sample_settings = Settings(data_dir=Path(temporary), ai_mode="mock", gemini_api_key=None)
        for module in (config_module, providers_module, services_module, workflow_module):
            module.settings = sample_settings
        sample_settings.ensure_directories()

        repository = Repository(sample_settings.app_db)
        workflow = Workflow(repository)
        service = ProjectService(repository, workflow.ai)
        project = repository.create_project(ProjectCreate(
            title="The Letter — Engineering Sample",
            source_text=FIXTURE.read_text(encoding="utf-8"),
            culture_id="maidani_mewari", locality="Rajsamand plains", setting="rural",
            period="contemporary_2020_2026", output_script="devanagari",
        ))
        project_id = project["id"]

        workflow.start(project_id)
        repaired_extraction = service.repair_continuity(project_id)
        workflow.resume(project_id, {"action": "approve", "revision_id": repaired_extraction["id"]})
        workflow.resume(project_id, {"action": "approve"})
        workflow.resume(project_id, {"action": "approve"})
        workflow.resume(project_id, {"action": "approve"})

        for asset in repository.list_assets(project_id):
            if asset["kind"] == "character_sheet":
                service.generate_asset(asset["id"])
                service.approve_asset(asset["id"])

        workflow.resume(project_id, {"action": "approve"})
        extraction = repository.latest_revision(project_id, "extraction")["payload"]
        for scene in extraction["scenes"]:
            asset = next(
                item for item in repository.list_assets(project_id)
                if item["kind"] == "scene_keyframe" and item["scene_id"] == scene["id"]
                and item["status"] != "invalidated"
            )
            generated = service.generate_asset(asset["id"])
            service.approve_scene_asset(generated["id"], SceneVisualApprovalRequest(
                geometry_notes="Mock sample geometry; replace through live visual review.",
                promote_as_set_reference=True,
            ))

        archive = ExportService(repository).build(project_id)
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        OUTPUT.mkdir(parents=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(OUTPUT)
        workflow._checkpoint_connection.close()

    print(f"Generated deterministic sample at {OUTPUT}")


if __name__ == "__main__":
    main()
