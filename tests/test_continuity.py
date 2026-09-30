from backend.continuity import check_continuity
from backend.providers import MockProvider
from backend.schemas import ContinuityEvent, ProductionElement, StateSnapshot, StateTransition
from backend.state_ledger import build_state_ledger


def test_detects_prop_and_costume_discontinuity():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(
        project_id,
        "Test",
        "INT. ROOM - DAY\n\nAMAR\nI have the letter.\n\nEXT. ROAD - DAY\n\nAMAR\nI am here.",
    )
    character_id = source.characters[0].id
    first, second = source.scenes
    source.state_transitions = [
        StateTransition(
            character_id=character_id, scene_id=first.id,
            before=StateSnapshot(character_id=character_id, scene_id=first.id, costume_id="costume-a"),
            events=["Takes letter"],
            after=StateSnapshot(character_id=character_id, scene_id=first.id, costume_id="costume-a", carries=["letter"]),
        ),
        StateTransition(
            character_id=character_id, scene_id=second.id,
            before=StateSnapshot(character_id=character_id, scene_id=second.id, costume_id="costume-b"),
            after=StateSnapshot(character_id=character_id, scene_id=second.id, costume_id="costume-b"),
        ),
    ]
    codes = {issue.code for issue in check_continuity(source)}
    assert "prop_transfer_gap" in codes
    assert "costume_change_gap" in codes


def test_scene_order_must_be_contiguous():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(project_id, "Test", "INT. A - DAY\nAction.\n\nEXT. B - DAY\nAction.")
    source.scenes[1].number = 4
    assert any(issue.code == "scene_order" for issue in check_continuity(source))


def test_missing_scene_metadata_is_blocking():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(project_id, "Test", "INT. ROOM - DAY\nRain hits the window.")
    source.scenes[0].weather = None
    source.scenes[0].mood = None
    issue = next(item for item in check_continuity(source) if item.code == "missing_scene_metadata")
    assert issue.severity == "blocking"
    assert "weather" in (issue.actual or "")
    assert "mood" in (issue.actual or "")


def test_detects_unknown_ids_and_unexplained_injury_recovery():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(project_id, "Test", "INT. A - DAY\nAMAR\nWait.\nEXT. B - DAY\nAMAR\nGo.")
    character_id = source.characters[0].id
    first, second = source.scenes
    second.character_ids.append("unknown-character")
    source.state_transitions = [
        StateTransition(
            character_id=character_id, scene_id=first.id,
            before=StateSnapshot(character_id=character_id, scene_id=first.id),
            after=StateSnapshot(character_id=character_id, scene_id=first.id, injuries=["cut hand"]),
        ),
        StateTransition(
            character_id=character_id, scene_id=second.id,
            before=StateSnapshot(character_id=character_id, scene_id=second.id),
            after=StateSnapshot(character_id=character_id, scene_id=second.id),
        ),
    ]
    codes = {issue.code for issue in check_continuity(source)}
    assert {"unknown_character", "injury_disappears"}.issubset(codes)


def test_event_ledger_carries_props_and_knowledge_without_false_loss():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(
        project_id, "Event ledger",
        "INT. A - DAY\nMARA\nWait.\n\nINT. B - DAY\nMARA\nRead this.\n\nEXT. C - DAY\nMARA\nKeep the copy.",
    )
    character_id = source.characters[0].id
    first, second, third = source.scenes
    key = ProductionElement(id="key", kind="prop", name="Brass key", scene_ids=[second.id, third.id])
    original = ProductionElement(id="original", kind="prop", name="Original log", scene_ids=[second.id])
    copied = ProductionElement(id="copy", kind="prop", name="Copied log", scene_ids=[third.id])
    source.production_elements.extend([key, original, copied])
    source.continuity_events = [
        ContinuityEvent(
            id="event-1", scene_id=first.id, character_id=character_id, kind="no_change",
            description="No durable change", evidence_block_ids=[first.blocks[0].id],
        ),
        ContinuityEvent(
            id="event-2", scene_id=second.id, character_id=character_id,
            kind="first_observed_prop", timing="first_observed", prop_id=key.id,
            description="The key is first seen at Mara's waist.", evidence_block_ids=[second.blocks[0].id],
        ),
        ContinuityEvent(
            id="event-3", scene_id=second.id, character_id=character_id,
            kind="acquire_prop", prop_id=original.id,
            description="Mara keeps the original log.", evidence_block_ids=[second.blocks[0].id],
        ),
        ContinuityEvent(
            id="event-4", scene_id=second.id, character_id=character_id,
            kind="learn_fact", fact="The delivery is legitimate",
            description="Mara verifies the delivery.", evidence_block_ids=[second.blocks[0].id],
        ),
        ContinuityEvent(
            id="event-5", scene_id=third.id, character_id=character_id,
            kind="release_prop", timing="before_scene", prop_id=original.id,
            description="The original is returned between scenes.", evidence_block_ids=[third.blocks[0].id],
        ),
        ContinuityEvent(
            id="event-6", scene_id=third.id, character_id=character_id,
            kind="derive_prop", timing="before_scene", prop_id=copied.id, related_prop_id=original.id,
            description="Mara receives a copy of the log.", evidence_block_ids=[third.blocks[0].id],
        ),
    ]
    transitions, ledger_issues = build_state_ledger(source)
    assert not [item for item in ledger_issues if item.severity == "blocking"]
    assert not [item for item in ledger_issues if item.code == "missing_state_coverage"]
    third_state = next(
        item for item in transitions if item.character_id == character_id and item.scene_id == third.id
    )
    assert set(third_state.before.carries) == {key.id, copied.id}
    assert third_state.before.knows == ["The delivery is legitimate"]
    source.state_transitions = transitions
    codes = {item.code for item in check_continuity(source)}
    assert "prop_transfer_gap" not in codes
    assert "knowledge_loss" not in codes


def test_event_ledger_reports_missing_appearance_coverage():
    project_id = "12345678-1234-5678-1234-567812345678"
    source = MockProvider().extract(project_id, "Coverage", "INT. ROOM - DAY\nMARA\nWait.")
    source.continuity_events = [ContinuityEvent(
        id="unrelated", scene_id=source.scenes[0].id, character_id="unknown-character",
        kind="no_change", description="Invalid event",
    )]
    issues = check_continuity(source)
    assert any(item.code == "invalid_continuity_event_reference" and item.severity == "blocking" for item in issues)
    assert any(item.code == "missing_state_coverage" for item in issues)
