from __future__ import annotations

from typing import Any

from backend.cultures.models import CultureRuntimeContext, ScriptSpec
from backend.schemas import AdaptedScreenplay, LanguageAudit


SCRIPT_ISSUE_CODE = "output_script_mismatch"
LANGUAGE_ISSUE_CODES = {
    "language_audit_incomplete",
    "unsupported_fallback_dialogue",
    "unapproved_variety_mixing",
    "language_audit_failed",
}


def validate_output_script(screenplay: AdaptedScreenplay, script: ScriptSpec) -> list[dict[str, Any]]:
    policy = script.validation
    if policy is None:
        return []
    ranges = [(int(item.start, 16), int(item.end, 16)) for item in policy.unicode_ranges]
    enforced_types = set(policy.enforced_block_types)
    issues: list[dict[str, Any]] = []
    for scene_number, scene in enumerate(screenplay.scenes, 1):
        for block in scene.blocks:
            if block.type.value not in enforced_types:
                continue
            letters = [character for character in block.adapted_text if character.isalpha()]
            if len(letters) < policy.minimum_letters_to_check:
                continue
            target_letters = sum(
                any(start <= ord(character) <= end for start, end in ranges)
                for character in letters
            )
            ratio = target_letters / len(letters)
            if ratio >= policy.minimum_target_letter_ratio:
                continue
            preview = " ".join(block.adapted_text.split())[:120]
            issues.append({
                "severity": "blocking",
                "code": SCRIPT_ISSUE_CODE,
                "scene_id": scene.source_scene_id,
                "scene_number": scene_number,
                "block_id": block.id,
                "block_type": block.type.value,
                "expected_script": script.display_name,
                "target_letter_ratio": round(ratio, 3),
                "message": (
                    f"{block.type.value.title()} block {block.id[:8]} in scene {scene_number} "
                    f"uses predominantly another writing script; expected {script.display_name}. Preview: {preview}"
                ),
            })
    return issues


def language_audit_issues(
    screenplay: AdaptedScreenplay, audit: LanguageAudit, context: CultureRuntimeContext,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    screenplay_scene_ids = {scene.source_scene_id for scene in screenplay.scenes}
    unknown_scene_ids = sorted({item.scene_id for item in audit.scenes} - screenplay_scene_ids)
    if unknown_scene_ids:
        issues.append({
            "severity": "blocking", "code": "language_audit_incomplete",
            "message": f"Language audit referenced unknown scenes: {unknown_scene_ids}.",
        })
    audit_by_scene = {item.scene_id: item for item in audit.scenes}
    for scene in screenplay.scenes:
        expected_dialogue_ids = {
            block.id for block in scene.blocks if block.type.value == "dialogue"
        }
        scene_audit = audit_by_scene.get(scene.source_scene_id)
        if not scene_audit:
            if expected_dialogue_ids:
                issues.append({
                    "severity": "blocking", "code": "language_audit_incomplete",
                    "scene_id": scene.source_scene_id,
                    "message": f"Language audit omitted speaking scene {scene.source_scene_id}.",
                })
            continue
        declared = set(scene_audit.dialogue_block_ids)
        classified_sequence = [
            *scene_audit.supported_target_variety_block_ids,
            *scene_audit.unsupported_fallback_language_block_ids,
            *scene_audit.mixed_or_unapproved_variety_block_ids,
        ]
        classified = set(classified_sequence)
        missing = sorted(expected_dialogue_ids - classified)
        unknown = sorted((declared | classified) - expected_dialogue_ids)
        duplicates = sorted({
            block_id for block_id in classified_sequence if classified_sequence.count(block_id) > 1
        })
        if declared != expected_dialogue_ids or missing or unknown or duplicates:
            issues.append({
                "severity": "blocking", "code": "language_audit_incomplete",
                "scene_id": scene.source_scene_id,
                "message": (
                    f"Language audit for scene {scene.source_scene_id} is incomplete or inconsistent. "
                    f"Missing={missing}; unknown={unknown}; multiply classified={duplicates}."
                ),
            })
        if scene_audit.unsupported_fallback_language_block_ids:
            fallback = (
                context.language_policy.fallback_language
                if context.language_policy and context.language_policy.fallback_language
                else "fallback language"
            )
            issues.append({
                "severity": "blocking", "code": "unsupported_fallback_dialogue",
                "scene_id": scene.source_scene_id,
                "message": (
                    f"Scene {scene.source_scene_id} contains unsupported {fallback} dialogue rather than "
                    f"{context.target_variety}: {scene_audit.unsupported_fallback_language_block_ids}"
                ),
            })
        if scene_audit.mixed_or_unapproved_variety_block_ids:
            issues.append({
                "severity": "blocking", "code": "unapproved_variety_mixing",
                "scene_id": scene.source_scene_id,
                "message": (
                    f"Scene {scene.source_scene_id} contains unsupported, mixed or confusable language forms: "
                    f"{scene_audit.mixed_or_unapproved_variety_block_ids}"
                ),
            })
    if not audit.passed and not any(
        issue["code"] in {"unsupported_fallback_dialogue", "unapproved_variety_mixing"}
        for issue in issues
    ):
        issues.append({"severity": "blocking", "code": "language_audit_failed", "message": audit.summary})
    return issues
