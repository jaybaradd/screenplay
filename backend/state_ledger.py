from __future__ import annotations

import uuid
from collections import defaultdict

from backend.schemas import ContinuityEvent, ContinuityIssue, SourceScreenplay, StateSnapshot, StateTransition


PROP_EVENT_KINDS = {
    "first_observed_prop", "acquire_prop", "release_prop", "transfer_prop", "derive_prop",
}


def _issue_id(code: str, entity_id: str | None, scenes: list[str], suffix: str = "") -> str:
    key = f"{code}:{entity_id or '-'}:{','.join(scenes)}:{suffix}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _snapshot(state: StateSnapshot, scene_id: str) -> StateSnapshot:
    value = state.model_copy(deep=True)
    value.scene_id = scene_id
    return value


def _default_state(character_id: str, scene_id: str) -> StateSnapshot:
    return StateSnapshot(character_id=character_id, scene_id=scene_id)


def _append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def _remove(values: list[str], value: str | None) -> bool:
    if value and value in values:
        values.remove(value)
        return True
    return False


def _validate_event(
    event: ContinuityEvent,
    scene_ids: set[str],
    character_ids: set[str],
    element_ids: set[str],
    block_scenes: dict[str, str],
) -> list[ContinuityIssue]:
    issues: list[ContinuityIssue] = []
    affected = [event.scene_id]
    unknown: list[str] = []
    if event.scene_id not in scene_ids:
        unknown.append(f"scene_id={event.scene_id}")
    if event.character_id not in character_ids:
        unknown.append(f"character_id={event.character_id}")
    if event.counterparty_character_id and event.counterparty_character_id not in character_ids:
        unknown.append(f"counterparty_character_id={event.counterparty_character_id}")
    if event.relationship_character_id and event.relationship_character_id not in character_ids:
        unknown.append(f"relationship_character_id={event.relationship_character_id}")
    for field, value in (("prop_id", event.prop_id), ("related_prop_id", event.related_prop_id)):
        if value and value not in element_ids:
            unknown.append(f"{field}={value}")
    invalid_blocks = [
        block_id for block_id in event.evidence_block_ids
        if block_id not in block_scenes or block_scenes[block_id] != event.scene_id
    ]
    if invalid_blocks:
        unknown.append(f"evidence_block_ids={invalid_blocks}")
    if unknown:
        issues.append(ContinuityIssue(
            id=_issue_id("invalid_continuity_event_reference", event.character_id, affected, event.id),
            severity="blocking", code="invalid_continuity_event_reference",
            entity_id=event.character_id, affected_scene_ids=affected,
            actual="; ".join(unknown),
            message="A continuity event references an unknown or out-of-scene canonical ID.",
            suggested_resolution="Correct the event references using the approved scene, character, prop and block IDs.",
            evidence_block_ids=event.evidence_block_ids,
            probable_cause="The extracted event did not use the supplied canonical records.",
        ))
    required_error = None
    if event.kind in PROP_EVENT_KINDS and not event.prop_id:
        required_error = "prop_id is required"
    elif event.kind == "first_observed_prop" and event.timing != "first_observed":
        required_error = "first_observed_prop must use timing=first_observed"
    elif event.timing == "first_observed" and event.kind != "first_observed_prop":
        required_error = "timing=first_observed is reserved for first_observed_prop"
    elif event.kind == "transfer_prop" and not event.counterparty_character_id:
        required_error = "counterparty_character_id is required"
    elif event.kind == "transfer_prop" and event.counterparty_character_id == event.character_id:
        required_error = "transfer counterparty must be a different character"
    elif event.kind == "derive_prop" and not event.related_prop_id:
        required_error = "related_prop_id is required"
    elif event.kind == "derive_prop" and event.prop_id == event.related_prop_id:
        required_error = "a derived physical item must have a distinct canonical prop ID"
    elif event.kind in {"learn_fact", "forget_fact", "injury", "recovery"} and not event.fact:
        required_error = "fact is required"
    elif event.kind in {"emotion_change", "relationship_change"} and not event.value:
        required_error = "value is required"
    elif event.kind == "relationship_change" and not event.relationship_character_id:
        required_error = "relationship_character_id is required"
    elif event.kind == "costume_change" and not (event.prop_id or event.value):
        required_error = "prop_id or value is required"
    elif event.kind != "no_change" and not event.evidence_block_ids:
        required_error = "at least one supporting source block is required"
    if required_error:
        issues.append(ContinuityIssue(
            id=_issue_id("incomplete_continuity_event", event.character_id, affected, event.id),
            severity="blocking", code="incomplete_continuity_event",
            entity_id=event.character_id, affected_scene_ids=affected,
            actual=required_error, message=f"Continuity event {event.id} is incomplete.",
            suggested_resolution="Complete the required event fields or remove the unsupported event.",
            evidence_block_ids=event.evidence_block_ids,
        ))
    return issues


def _apply_event(
    event: ContinuityEvent,
    states: dict[str, StateSnapshot],
    scene_id: str,
) -> ContinuityIssue | None:
    state = states.setdefault(event.character_id, _default_state(event.character_id, scene_id))
    precondition_issue: ContinuityIssue | None = None
    if event.kind in {"first_observed_prop", "acquire_prop"}:
        _append_unique(state.carries, event.prop_id)
    elif event.kind == "release_prop":
        if not _remove(state.carries, event.prop_id):
            precondition_issue = _precondition_issue(event, "released prop was not previously established as carried")
    elif event.kind == "transfer_prop":
        if not _remove(state.carries, event.prop_id):
            precondition_issue = _precondition_issue(event, "transferred prop was not previously established as carried")
        if event.counterparty_character_id:
            recipient = states.setdefault(
                event.counterparty_character_id,
                _default_state(event.counterparty_character_id, scene_id),
            )
            _append_unique(recipient.carries, event.prop_id)
    elif event.kind == "derive_prop":
        _append_unique(state.carries, event.prop_id)
    elif event.kind == "learn_fact":
        _append_unique(state.knows, event.fact)
    elif event.kind == "forget_fact":
        if not _remove(state.knows, event.fact):
            precondition_issue = _precondition_issue(event, "forgotten fact was not previously established as known")
    elif event.kind == "costume_change":
        state.costume_id = event.prop_id or event.value
    elif event.kind == "injury":
        _append_unique(state.injuries, event.fact)
    elif event.kind == "recovery":
        if not _remove(state.injuries, event.fact):
            precondition_issue = _precondition_issue(event, "recovered injury was not previously established")
    elif event.kind == "emotion_change":
        state.emotion = event.value
    elif event.kind == "relationship_change" and event.relationship_character_id:
        state.relationship_state[event.relationship_character_id] = event.value or ""
    return precondition_issue


def _precondition_issue(event: ContinuityEvent, detail: str) -> ContinuityIssue:
    return ContinuityIssue(
        id=_issue_id("continuity_event_precondition", event.character_id, [event.scene_id], event.id),
        severity="warning", code="continuity_event_precondition", entity_id=event.character_id,
        affected_scene_ids=[event.scene_id], actual=detail,
        message="A continuity event is supported by the scene but its incoming state was not established.",
        suggested_resolution="Review earlier event coverage; keep this event if the source intentionally reveals prior state late.",
        evidence_block_ids=event.evidence_block_ids,
        probable_cause="An earlier appearance may be missing a first-observation or acquisition event.",
    )


def build_state_ledger(screenplay: SourceScreenplay) -> tuple[list[StateTransition], list[ContinuityIssue]]:
    """Derive complete character snapshots from evidence-linked continuity events."""
    scene_ids = {scene.id for scene in screenplay.scenes}
    character_ids = {character.id for character in screenplay.characters}
    element_ids = {element.id for element in screenplay.production_elements}
    block_scenes = {block.id: scene.id for scene in screenplay.scenes for block in scene.blocks}
    issues: list[ContinuityIssue] = []
    seen_event_ids: set[str] = set()
    valid_events: list[ContinuityEvent] = []
    for event in screenplay.continuity_events:
        if event.id in seen_event_ids:
            issues.append(ContinuityIssue(
                id=_issue_id("duplicate_continuity_event", event.character_id, [event.scene_id], event.id),
                severity="blocking", code="duplicate_continuity_event", entity_id=event.character_id,
                affected_scene_ids=[event.scene_id], message=f"Continuity event ID {event.id} is duplicated.",
                suggested_resolution="Keep one canonical event record for this occurrence.",
            ))
            continue
        seen_event_ids.add(event.id)
        event_issues = _validate_event(event, scene_ids, character_ids, element_ids, block_scenes)
        issues.extend(event_issues)
        if not any(item.severity == "blocking" for item in event_issues):
            valid_events.append(event)

    events_by_scene: dict[str, list[ContinuityEvent]] = defaultdict(list)
    for event in valid_events:
        events_by_scene[event.scene_id].append(event)

    states: dict[str, StateSnapshot] = {}
    transitions: list[StateTransition] = []
    for scene in sorted(screenplay.scenes, key=lambda item: item.number):
        scene_events = events_by_scene.get(scene.id, [])
        covered_characters = {
            character_id
            for event in scene_events
            for character_id in (event.character_id, event.counterparty_character_id)
            if character_id
        }
        for character_id in scene.character_ids:
            if character_id not in covered_characters:
                issues.append(ContinuityIssue(
                    id=_issue_id("missing_state_coverage", character_id, [scene.id]),
                    severity="warning", code="missing_state_coverage", entity_id=character_id,
                    affected_scene_ids=[scene.id],
                    message="A character appears in this scene without a continuity event or explicit no-change record.",
                    suggested_resolution="Add evidence-backed events, or add a no-change event after reviewing the scene.",
                    probable_cause="The continuity-only extraction skipped this character appearance.",
                ))
        for character_id in scene.character_ids:
            states.setdefault(character_id, _default_state(character_id, scene.id))

        pre_scene_events = [item for item in scene_events if item.timing in {"before_scene", "first_observed"}]
        during_scene_events = [item for item in scene_events if item.timing == "during_scene"]
        for event in pre_scene_events:
            issue = _apply_event(event, states, scene.id)
            if issue:
                issues.append(issue)
        before = {
            character_id: _snapshot(states[character_id], scene.id)
            for character_id in scene.character_ids
        }
        for event in during_scene_events:
            issue = _apply_event(event, states, scene.id)
            if issue:
                issues.append(issue)
        for character_id in scene.character_ids:
            relevant = [
                item.description for item in scene_events
                if character_id in {item.character_id, item.counterparty_character_id}
            ]
            transitions.append(StateTransition(
                character_id=character_id, scene_id=scene.id,
                before=before[character_id], events=relevant,
                after=_snapshot(states[character_id], scene.id),
            ))
    return transitions, issues
