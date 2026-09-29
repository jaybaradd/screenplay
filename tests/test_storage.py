from pathlib import Path

from backend.schemas import ProjectCreate
from backend.storage import Repository


def test_versioned_revisions_and_idempotency(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Test", source_text="A" * 120, locality="Rajsamand plains", setting="rural",
    ))
    first = repository.create_revision(project["id"], "example", {"value": 1})
    second = repository.create_revision(project["id"], "example", {"value": 2}, first["id"])
    assert first["version"] == 1
    assert second["version"] == 2
    assert repository.latest_revision(project["id"], "example")["payload"] == {"value": 2}
    repository.save_idempotent("scope", "key", {"revision_id": second["id"]})
    assert repository.get_idempotent("scope", "key")["revision_id"] == second["id"]


def test_asset_identity_deduplicates_and_supersedes_old_dependencies(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Assets", source_text="A" * 120, locality="Rajsamand plains", setting="rural",
    ))
    first = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "prompt", "hash-a")
    repeated = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "prompt", "hash-a")
    assert repeated["id"] == first["id"]
    replacement = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "new prompt", "hash-b")
    assert replacement["id"] != first["id"]
    assets = {item["id"]: item for item in repository.list_assets(project["id"])}
    assert assets[first["id"]]["status"] == "invalidated"
    assert assets[replacement["id"]]["status"] == "pending"
