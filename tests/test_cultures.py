from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.cultures.context import CultureContextCompiler
from backend.cultures.models import TargetSelection
from backend.cultures.registry import CultureRegistry, culture_registry
from backend.prompts import adaptation_prompt, cultural_research_prompt, language_audit_prompt, layer_prompt, visual_manifest_prompt
from backend.schemas import ProjectCreate
from backend.schemas import RecordPatch
from backend.services import ProjectService
from backend.storage import Repository


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "cultures"
FORBIDDEN_LEAKS = ("maidani", "mewari", "rajasthan", "marwari", "hindi", "devanagari")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_production_registry_exposes_only_enabled_profiles():
    summaries = culture_registry.summaries()
    assert [item.culture_id for item in summaries] == ["maidani_mewari"]
    snapshot = culture_registry.snapshot("maidani_mewari")
    assert snapshot.profile.language_policy.minimum_supported_features == 6
    assert snapshot.profile.supported_scripts[0].id == "devanagari"
    assert {item.casefold() for item in snapshot.profile.commonly_confused_languages} >= {
        "marwari", "standard hindi",
    }
    assert any("palaces" in item.text for item in snapshot.profile.constraints)
    assert len(snapshot.profile_hash) == 64


def test_generic_python_modules_contain_no_production_culture_literals():
    paths = [
        *sorted((REPOSITORY_ROOT / "backend").rglob("*.py")),
        *sorted((REPOSITORY_ROOT / "frontend").rglob("*.py")),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        for forbidden in FORBIDDEN_LEAKS:
            assert forbidden not in text, f"{forbidden!r} leaked into generic module {path}"


def test_fictional_profile_proves_prompt_isolation():
    registry = CultureRegistry(FIXTURE_ROOT)
    assert registry.summaries() == []
    snapshot = registry.snapshot("coastal_asteri")
    selection = TargetSelection(
        culture_id="coastal_asteri", locality="South Harbour", setting="port",
        period="asteri_present", output_script="asteri_script",
    )
    context = CultureContextCompiler(snapshot, selection).for_research()
    visual_context = CultureContextCompiler(snapshot, selection).for_visual_manifest()
    assert visual_context.language_policy is None
    assert visual_context.language_guide is None
    assert visual_context.confusable_languages == []
    assert all(item["kind"] != "language_boundary" for item in visual_context.constraints)
    assert len({item["id"] for item in visual_context.constraints}) == len(visual_context.constraints)
    repeated_policy = snapshot.profile.constraints[-1].model_dump(mode="json")
    repeated_policy["origin"] = "profile_policy"
    deduplicated = CultureContextCompiler(
        snapshot, selection, {"constraints": [repeated_policy]},
    ).for_visual_manifest()
    assert [item["id"] for item in deduplicated.constraints].count(repeated_policy["id"]) == 1
    prompts = [
        cultural_research_prompt(context),
        layer_prompt("visual_world", {}, CultureContextCompiler(snapshot, selection).for_layer("visual_world")),
        adaptation_prompt(
            {"blocks": []}, {"story_contract": {}}, {"claims": []}, {"layers": []}, context,
        ),
        language_audit_prompt({"scenes": []}, {"claims": []}, context),
        visual_manifest_prompt({}, {}, {}, visual_context, [], []),
    ]
    combined = "\n".join(prompts).casefold()
    for forbidden in FORBIDDEN_LEAKS:
        assert forbidden not in combined
    assert "harbour asteri" in combined
    assert "north asteri" in combined


def test_profile_hash_is_stable_and_installed_changes_do_not_mutate_snapshot(tmp_path: Path):
    profile_root = tmp_path / "cultures"
    shutil.copytree(FIXTURE_ROOT, profile_root)
    registry = CultureRegistry(profile_root)
    frozen = registry.snapshot("coastal_asteri")
    assert frozen.profile_hash == CultureRegistry(profile_root).snapshot("coastal_asteri").profile_hash

    profile_path = profile_root / "coastal_asteri" / "profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["version"] = "9.4.2"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    registry.reload()
    assert registry.snapshot("coastal_asteri").profile_hash != frozen.profile_hash
    assert frozen.profile.version == "9.4.1"

    repository = Repository(tmp_path / "app.sqlite")
    project = repository.create_project(ProjectCreate(
        title="Frozen profile", source_text="A" * 120, culture_id="coastal_asteri",
        locality="South Harbour", setting="port", period="asteri_present", output_script="asteri_script",
    ), frozen)
    saved = repository.latest_revision(project["id"], "culture_profile_snapshot")["payload"]
    assert saved["profile_hash"] == frozen.profile_hash
    assert saved["profile"]["version"] == "9.4.1"
    service = ProjectService(repository, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="immutable"):
        service.patch_record(project["id"], RecordPatch(
            document_kind="culture_profile_snapshot", field_path="/profile/version", value="1.0.0",
        ))


def test_selection_and_font_contracts_are_enforced(tmp_path: Path):
    registry = CultureRegistry(FIXTURE_ROOT)
    with pytest.raises(ValueError, match="Unsupported setting"):
        registry.validate_selection(TargetSelection(
            culture_id="coastal_asteri", locality="South Harbour", setting="mountain",
            period="asteri_present", output_script="asteri_script",
        ))

    profile_root = tmp_path / "cultures"
    shutil.copytree(FIXTURE_ROOT, profile_root)
    profile_path = profile_root / "coastal_asteri" / "profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["supported_scripts"][0]["font_path"] = "assets/fonts/missing.ttf"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or unsafe font"):
        CultureRegistry(profile_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(version="not-semver"), "version"),
        (
            lambda payload: payload["supported_scripts"].append(payload["supported_scripts"][0].copy()),
            "Duplicate script IDs",
        ),
        (
            lambda payload: payload["required_research_dimensions"].append("unsupported_dimension"),
            "Unsupported research dimensions",
        ),
    ],
)
def test_malformed_profile_contracts_are_rejected(tmp_path: Path, mutation, message: str):
    profile_root = tmp_path / "cultures"
    shutil.copytree(FIXTURE_ROOT, profile_root)
    profile_path = profile_root / "coastal_asteri" / "profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    mutation(payload)
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match=message):
        CultureRegistry(profile_root)
