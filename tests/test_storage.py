import sqlite3
from pathlib import Path

import pytest

from backend.schemas import AssetStatus, ProjectCreate
from backend.storage import Repository


def test_pre_profile_database_is_rejected_without_deletion(tmp_path: Path):
    database = tmp_path / "app.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, culture TEXT NOT NULL)")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="predates culture profiles"):
        Repository(database)
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT name FROM sqlite_master WHERE name='projects'").fetchone()
    connection.close()


def test_versioned_revisions_and_idempotency(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Test", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
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
        title="Assets", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural", period="contemporary_2020_2026", output_script="devanagari",
    ))
    first = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "prompt", "hash-a")
    repeated = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "prompt", "hash-a")
    assert repeated["id"] == first["id"]
    replacement = repository.upsert_asset(project["id"], "character_sheet", "appearance-1", "new prompt", "hash-b")
    assert replacement["id"] != first["id"]
    assets = {item["id"]: item for item in repository.list_assets(project["id"])}
    assert assets[first["id"]]["status"] == "invalidated"
    assert assets[replacement["id"]]["status"] == "pending"


def test_recompiling_retained_dependency_reactivates_it_without_losing_image(tmp_path: Path):
    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Recompile", source_text="A" * 120, culture_id="maidani_mewari",
        locality="Rajsamand plains", setting="rural",
        period="contemporary_2020_2026", output_script="devanagari",
    ))
    retained_image = tmp_path / "retained-scene.png"
    retained_image.write_bytes(b"retained-scene-image")
    original = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-2", "Compiled continuity prompt",
        "approved-continuity-hash", scene_id="scene-2",
    )
    repository.update_asset(
        original["id"], AssetStatus.generated, path=str(retained_image), increment_attempt=True,
    )
    correction = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-2", "Corrected scene prompt",
        "correction-hash", scene_id="scene-2",
    )
    assert repository.get_asset(original["id"])["status"] == "invalidated"

    recompiled = repository.upsert_asset(
        project["id"], "scene_keyframe", "scene-visual-2", "Compiled continuity prompt",
        "approved-continuity-hash", scene_id="scene-2",
    )

    assert recompiled["id"] == original["id"]
    assert recompiled["status"] == "pending"
    assert recompiled["path"] == str(retained_image)
    assert recompiled["attempt"] == 1
    assert repository.get_asset(correction["id"])["status"] == "invalidated"
