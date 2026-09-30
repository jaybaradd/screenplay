from backend.continuity import check_continuity
from backend.providers import MockProvider
from backend.schemas import StateSnapshot, StateTransition


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
