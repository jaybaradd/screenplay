from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Iterable

from backend.schemas import ContinuityIssue, SourceScreenplay, StateTransition
from backend.state_ledger import build_state_ledger


def issue_id(code: str, entity_id: str | None, scenes: Iterable[str]) -> str:
    key = f"{code}:{entity_id or '-'}:{','.join(scenes)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def check_continuity(screenplay: SourceScreenplay) -> list[ContinuityIssue]:
    issues: list[ContinuityIssue] = []
    expected_numbers = list(range(1, len(screenplay.scenes) + 1))
    actual_numbers = [scene.number for scene in screenplay.scenes]
    if actual_numbers != expected_numbers:
        scene_ids = [scene.id for scene in screenplay.scenes]
        issues.append(ContinuityIssue(
            id=issue_id("scene_order", None, scene_ids), severity="blocking", code="scene_order",
            affected_scene_ids=scene_ids, expected=str(expected_numbers), actual=str(actual_numbers),
            message="Scenes are missing, duplicated, or out of order.",
            suggested_resolution="Restore one ordered record for every source scene.",
        ))

    character_ids = {character.id for character in screenplay.characters}
    element_ids = {element.id for element in screenplay.production_elements}
    for scene in screenplay.scenes:
        required_metadata = {
            "int_ext": scene.int_ext,
            "location": scene.location,
            "sub_location": scene.sub_location,
            "time": scene.time,
            "day_or_date": scene.day_or_date,
            "weather": scene.weather,
            "mood": scene.mood,
            "summary": scene.summary,
            "dramatic_purpose": scene.dramatic_purpose,
        }
        missing_metadata = [
            field for field, value in required_metadata.items()
            if value is None or not str(value).strip()
        ]
        if missing_metadata:
            issues.append(ContinuityIssue(
                id=issue_id("missing_scene_metadata", scene.id, [scene.id]), severity="blocking",
                code="missing_scene_metadata", entity_id=scene.id, affected_scene_ids=[scene.id],
                expected="Every required scene field has a value or an explicit 'Not stated in source' marker.",
                actual=f"Missing fields: {', '.join(missing_metadata)}",
                message=f"Scene {scene.number} has incomplete extraction metadata: {', '.join(missing_metadata)}.",
                suggested_resolution="Extract an evidence-based value or enter 'Not stated in source'; do not invent facts.",
            ))
        for character_id in scene.character_ids:
            if character_id not in character_ids:
                issues.append(ContinuityIssue(
                    id=issue_id("unknown_character", character_id, [scene.id]), severity="blocking",
                    code="unknown_character", entity_id=character_id, affected_scene_ids=[scene.id],
                    message="Scene references an unknown character ID.",
                ))
        for element_id in scene.production_element_ids:
            if element_id not in element_ids:
                issues.append(ContinuityIssue(
                    id=issue_id("unknown_production_element", element_id, [scene.id]), severity="blocking",
                    code="unknown_production_element", entity_id=element_id, affected_scene_ids=[scene.id],
                    message="Scene references an unknown production-element ID.",
                ))

    transitions_to_check = screenplay.state_transitions
    event_based = bool(screenplay.continuity_events)
    if event_based:
        transitions_to_check, ledger_issues = build_state_ledger(screenplay)
        issues.extend(ledger_issues)

    scene_order = {scene.id: index for index, scene in enumerate(screenplay.scenes)}
    transitions_by_character: dict[str, list[StateTransition]] = defaultdict(list)
    for transition in transitions_to_check:
        transitions_by_character[transition.character_id].append(transition)
    for character_id, transitions in transitions_by_character.items():
        transitions.sort(key=lambda item: scene_order.get(item.scene_id, 10**9))
        for previous, current in zip(transitions, transitions[1:]):
            affected = [previous.scene_id, current.scene_id]
            lost_props = sorted(set(previous.after.carries) - set(current.before.carries))
            gained_props = sorted(set(current.before.carries) - set(previous.after.carries))
            if (lost_props or gained_props) and not event_based:
                issues.append(ContinuityIssue(
                    id=issue_id("prop_transfer_gap", character_id, affected), severity="blocking",
                    code="prop_transfer_gap", entity_id=character_id, affected_scene_ids=affected,
                    expected=f"Carries {previous.after.carries}", actual=f"Carries {current.before.carries}",
                    message="A carried prop changes between appearances without a recorded transfer.",
                    suggested_resolution="Add a transfer/loss event or correct the before-state.",
                    added_items=gained_props, removed_items=lost_props,
                    probable_cause="The extracted before/after snapshots disagree or an intervening scene was omitted.",
                ))
            if previous.after.costume_id != current.before.costume_id and not event_based:
                issues.append(ContinuityIssue(
                    id=issue_id("costume_change_gap", character_id, affected), severity="blocking",
                    code="costume_change_gap", entity_id=character_id, affected_scene_ids=affected,
                    expected=previous.after.costume_id, actual=current.before.costume_id,
                    message="Costume changes between appearances without a recorded reason.",
                    suggested_resolution="Record a justified costume change or keep the prior costume.",
                ))
            healed = sorted(set(previous.after.injuries) - set(current.before.injuries))
            if healed and not event_based:
                issues.append(ContinuityIssue(
                    id=issue_id("injury_disappears", character_id, affected), severity="blocking",
                    code="injury_disappears", entity_id=character_id, affected_scene_ids=affected,
                    expected=str(previous.after.injuries), actual=str(current.before.injuries),
                    message="An injury disappears without a recovery event.",
                ))
    return issues


def blocking_issues(issues: list[ContinuityIssue]) -> list[ContinuityIssue]:
    return [issue for issue in issues if issue.severity == "blocking"]
