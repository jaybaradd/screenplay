from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Iterable

from backend.schemas import ContinuityIssue, SourceScreenplay, StateTransition


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

    scene_order = {scene.id: index for index, scene in enumerate(screenplay.scenes)}
    transitions_by_character: dict[str, list[StateTransition]] = defaultdict(list)
    for transition in screenplay.state_transitions:
        transitions_by_character[transition.character_id].append(transition)
    for character_id, transitions in transitions_by_character.items():
        transitions.sort(key=lambda item: scene_order.get(item.scene_id, 10**9))
        for previous, current in zip(transitions, transitions[1:]):
            affected = [previous.scene_id, current.scene_id]
            lost_props = sorted(set(previous.after.carries) - set(current.before.carries))
            gained_props = sorted(set(current.before.carries) - set(previous.after.carries))
            if lost_props or gained_props:
                issues.append(ContinuityIssue(
                    id=issue_id("prop_transfer_gap", character_id, affected), severity="blocking",
                    code="prop_transfer_gap", entity_id=character_id, affected_scene_ids=affected,
                    expected=f"Carries {previous.after.carries}", actual=f"Carries {current.before.carries}",
                    message="A carried prop changes between appearances without a recorded transfer.",
                    suggested_resolution="Add a transfer/loss event or correct the before-state.",
                ))
            if previous.after.costume_id != current.before.costume_id:
                issues.append(ContinuityIssue(
                    id=issue_id("costume_change_gap", character_id, affected), severity="blocking",
                    code="costume_change_gap", entity_id=character_id, affected_scene_ids=affected,
                    expected=previous.after.costume_id, actual=current.before.costume_id,
                    message="Costume changes between appearances without a recorded reason.",
                    suggested_resolution="Record a justified costume change or keep the prior costume.",
                ))
            missing_knowledge = sorted(set(previous.after.knows) - set(current.before.knows))
            if missing_knowledge:
                issues.append(ContinuityIssue(
                    id=issue_id("knowledge_loss", character_id, affected), severity="warning",
                    code="knowledge_loss", entity_id=character_id, affected_scene_ids=affected,
                    expected=str(previous.after.knows), actual=str(current.before.knows),
                    message="Previously known information is absent from the next state.",
                ))
            healed = sorted(set(previous.after.injuries) - set(current.before.injuries))
            if healed:
                issues.append(ContinuityIssue(
                    id=issue_id("injury_disappears", character_id, affected), severity="blocking",
                    code="injury_disappears", entity_id=character_id, affected_scene_ids=affected,
                    expected=str(previous.after.injuries), actual=str(current.before.injuries),
                    message="An injury disappears without a recovery event.",
                ))
    return issues


def blocking_issues(issues: list[ContinuityIssue]) -> list[ContinuityIssue]:
    return [issue for issue in issues if issue.severity == "blocking"]

