from __future__ import annotations

import copy
import uuid

import streamlit as st

from frontend.client import client
from frontend.common import require_project, run_action, setup_page


setup_page("Scene Extraction", "🧩")
project = require_project()
st.title("Scene extraction and canonical records")

if project["stage"] in {"created", "extracting"}:
    label = "Extract screenplay" if project["stage"] == "created" else "Retry extraction"
    run_action(label, lambda: client.advance(project["id"]), success="Extraction completed; review the records.")
    st.stop()

revision = project["revisions"].get("extraction")
if not revision:
    st.info("Extraction is still pending.")
    st.stop()
payload = revision["payload"]
scene_names = {
    item["id"]: f"Scene {item['number']} — {item['heading']}" for item in payload["scenes"]
}
character_names = {item["id"]: item["name"] for item in payload["characters"]}
element_names = {item["id"]: item["name"] for item in payload["production_elements"]}
block_records = {
    block["id"]: (scene["id"], block["text"])
    for scene in payload["scenes"] for block in scene["blocks"]
}
issues = project["revisions"].get("continuity_issues", {}).get("payload", [])
blocking_issues = [item for item in issues if item["severity"] == "blocking"]
warning_issues = [item for item in issues if item["severity"] == "warning"]
can_edit = project["stage"] == "extraction_review"

if issues:
    st.warning(
        f"Review the Issues tab before approval: {len(blocking_issues)} blocking issue(s), "
        f"{len(warning_issues)} warning(s)."
    )
else:
    st.success("Extraction checks found no continuity issues.")
if not can_edit:
    st.info("This extraction revision is already approved and is read-only so downstream plans remain reproducible.")


def _restore_nested(rows, originals: dict[str, list[dict]], field: str) -> list[dict]:
    restored: list[dict] = []
    for row in rows:
        item = dict(row)
        item[field] = copy.deepcopy(originals[item["id"]])
        restored.append(item)
    return restored

tabs = st.tabs(["Scenes", "Characters", "Production", "State", "Story contract", "Issues"])
with tabs[0]:
    scene_rows = copy.deepcopy(payload["scenes"])
    scene_blocks = {item["id"]: item["blocks"] for item in payload["scenes"]}
    for item in scene_rows:
        item["blocks"] = f"{len(item['blocks'])} ordered source blocks"
    edited_scenes = st.data_editor(
        scene_rows, width="stretch", num_rows="fixed", key=f"scenes-{revision['id']}",
        disabled=True if not can_edit else ["id", "blocks"],
        column_config={"blocks": st.column_config.TextColumn("Blocks", help="Read-only source block count")},
    )
    if can_edit:
        run_action(
            "Save scene edits",
            lambda: client.patch(
                project["id"], "extraction", "/scenes",
                _restore_nested(edited_scenes, scene_blocks, "blocks"), revision["sha256"],
            ),
            key="save-scenes",
        )
with tabs[1]:
    character_rows = copy.deepcopy(payload["characters"])
    character_relationships = {item["id"]: item["relationships"] for item in payload["characters"]}
    for item in character_rows:
        rendered_relationships = []
        for relationship in item["relationships"]:
            target = character_names.get(relationship["target_character_id"], relationship["target_character_id"][:8])
            note = f" — {relationship['notes']}" if relationship.get("notes") else ""
            rendered_relationships.append(f"{relationship['label']} → {target}{note}")
        item["relationships"] = "\n".join(rendered_relationships) or "No explicit relationships extracted"
    edited_characters = st.data_editor(
        character_rows, width="stretch", num_rows="fixed", key=f"characters-{revision['id']}",
        disabled=True if not can_edit else ["id", "relationships"], row_height=76,
        column_config={
            "relationships": st.column_config.TextColumn(
                "Relationships", help="Readable canonical relationships; kept read-only in this grid."
            ),
        },
    )
    if can_edit:
        run_action(
            "Save character edits",
            lambda: client.patch(
                project["id"], "extraction", "/characters",
                _restore_nested(edited_characters, character_relationships, "relationships"), revision["sha256"],
            ),
            key="save-characters",
        )
    if can_edit and len(payload["characters"]) > 1:
        labels = {f"{item['name']} · {item['id'][:8]}": item["id"] for item in payload["characters"]}
        primary_label = st.selectbox("Canonical character", labels, key="merge-primary")
        duplicates = st.multiselect("Aliases / duplicate records to merge", [label for label in labels if label != primary_label])
        run_action(
            "Merge selected records",
            lambda: client.merge(project["id"], "character", labels[primary_label], [labels[item] for item in duplicates]),
            key="merge", disabled=not duplicates,
        )
with tabs[2]:
    edited_elements = st.data_editor(
        payload["production_elements"], width="stretch", num_rows="fixed", key=f"elements-{revision['id']}",
        disabled=True if not can_edit else ["id"],
    )
    if can_edit:
        run_action(
            "Save production edits",
            lambda: client.patch(
                project["id"], "extraction", "/production_elements", edited_elements, revision["sha256"],
            ),
            key="save-production",
        )
with tabs[3]:
    st.subheader("Continuity events")
    st.caption(
        "Events are the source-linked facts the application uses to derive before/after state. "
        "Repairing this ledger does not rewrite scenes, characters or dialogue."
    )
    if can_edit:
        run_action(
            "Repair continuity ledger only",
            lambda: client.repair_continuity(project["id"]),
            success="Continuity events repaired and deterministic state recalculated.",
            key="repair-continuity",
        )
    events = copy.deepcopy(payload.get("continuity_events", []))
    if events:
        rows = []
        for event in events:
            rows.append({
                "id": event["id"],
                "scene": scene_names.get(event["scene_id"], event["scene_id"]),
                "character": character_names.get(event["character_id"], event["character_id"]),
                "kind": event["kind"], "timing": event["timing"],
                "prop": element_names.get(event.get("prop_id"), event.get("prop_id") or ""),
                "description": event["description"], "confidence": event["confidence"],
            })
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("This extraction still uses legacy snapshots. Run the scoped repair to create evidence-linked events.")

    if can_edit:
        editor_modes = ["Edit an event", "Add an event"] if events else ["Add an event"]
        mode = st.radio("Event editor", editor_modes, horizontal=True)
        selected_event = None
        if mode == "Edit an event":
            event_labels = {
                f"{scene_names.get(item['scene_id'], item['scene_id'])} · "
                f"{character_names.get(item['character_id'], item['character_id'])} · {item['kind']} · {item['id'][:8]}": item
                for item in events
            }
            selected_label = st.selectbox("Event", list(event_labels), key=f"event-select-{revision['id']}")
            selected_event = event_labels[selected_label]
        default_scene = next(
            (item for item in payload["scenes"] if item["character_ids"]), payload["scenes"][0]
        )
        default_character_id = (
            default_scene["character_ids"][0] if default_scene["character_ids"] else payload["characters"][0]["id"]
        )
        base = selected_event or {
            "id": str(uuid.uuid4()), "scene_id": default_scene["id"],
            "character_id": default_character_id,
            "kind": "no_change", "timing": "during_scene", "prop_id": None,
            "related_prop_id": None, "counterparty_character_id": None, "fact": None,
            "value": None, "relationship_character_id": None, "evidence_block_ids": [],
            "description": "Reviewed: no durable continuity change.", "confidence": "medium",
        }
        scene_options = list(scene_names)
        scene_id = st.selectbox(
            "Scene", scene_options, index=scene_options.index(base["scene_id"]),
            format_func=lambda value: scene_names[value], key=f"event-scene-{revision['id']}-{mode}",
        )
        character_options = next(
            item["character_ids"] for item in payload["scenes"] if item["id"] == scene_id
        )
        if not character_options:
            character_options = list(character_names)
        character_id = st.selectbox(
            "Character", character_options,
            index=character_options.index(base["character_id"]) if base["character_id"] in character_options else 0,
            format_func=lambda value: character_names[value], key=f"event-character-{revision['id']}-{mode}",
        )
        event_kinds = [
            "no_change", "first_observed_prop", "acquire_prop", "release_prop", "transfer_prop",
            "derive_prop", "learn_fact", "forget_fact", "costume_change", "injury", "recovery",
            "emotion_change", "relationship_change",
        ]
        kind = st.selectbox(
            "Event type", event_kinds, index=event_kinds.index(base["kind"]),
            key=f"event-kind-{revision['id']}-{mode}",
        )
        if kind == "first_observed_prop":
            timing = "first_observed"
            st.caption("Timing: first observed — this reveals existing possession without inventing an acquisition.")
        else:
            timings = ["before_scene", "during_scene"]
            current_timing = base["timing"] if base["timing"] in timings else "during_scene"
            timing = st.selectbox(
                "Timing", timings, index=timings.index(current_timing),
                key=f"event-timing-{revision['id']}-{mode}",
            )
        nullable_characters = [None, *character_names]
        prop_options = [
            None, *[
                item["id"] for item in payload["production_elements"]
                if item["kind"] in {"prop", "costume", "grooming", "jewellery"}
            ],
        ]
        prop_id = None
        related_prop_id = None
        counterparty_id = None
        relationship_character_id = None
        fact = ""
        value = ""
        if kind in {
            "first_observed_prop", "acquire_prop", "release_prop", "transfer_prop",
            "derive_prop", "costume_change",
        }:
            prop_id = st.selectbox(
                "Prop / costume", prop_options,
                index=prop_options.index(base.get("prop_id")) if base.get("prop_id") in prop_options else 0,
                format_func=lambda selected: "None" if selected is None else element_names[selected],
                key=f"event-prop-{revision['id']}-{mode}",
            )
        if kind == "derive_prop":
            related_prop_id = st.selectbox(
                "Source prop for the derived item", prop_options,
                index=prop_options.index(base.get("related_prop_id")) if base.get("related_prop_id") in prop_options else 0,
                format_func=lambda selected: "None" if selected is None else element_names[selected],
                key=f"event-related-prop-{revision['id']}-{mode}",
            )
        if kind == "transfer_prop":
            counterparty_id = st.selectbox(
                "Transfer counterparty", nullable_characters,
                index=nullable_characters.index(base.get("counterparty_character_id")) if base.get("counterparty_character_id") in nullable_characters else 0,
                format_func=lambda selected: "None" if selected is None else character_names[selected],
                key=f"event-counterparty-{revision['id']}-{mode}",
            )
        if kind == "relationship_change":
            relationship_character_id = st.selectbox(
                "Relationship character", nullable_characters,
                index=nullable_characters.index(base.get("relationship_character_id")) if base.get("relationship_character_id") in nullable_characters else 0,
                format_func=lambda selected: "None" if selected is None else character_names[selected],
                key=f"event-relationship-character-{revision['id']}-{mode}",
            )
        if kind in {"learn_fact", "forget_fact", "injury", "recovery"}:
            fact = st.text_input(
                "Knowledge / injury fact", value=base.get("fact") or "",
                key=f"event-fact-{revision['id']}-{mode}",
            )
        if kind in {"costume_change", "emotion_change", "relationship_change"}:
            value = st.text_input(
                "New costume, emotion or relationship value", value=base.get("value") or "",
                key=f"event-value-{revision['id']}-{mode}",
            )
        scene_block_ids = [block["id"] for scene in payload["scenes"] if scene["id"] == scene_id for block in scene["blocks"]]
        evidence = st.multiselect(
            "Supporting source blocks", scene_block_ids,
            default=[item for item in base.get("evidence_block_ids", []) if item in scene_block_ids],
            format_func=lambda block_id: block_records[block_id][1][:100],
            key=f"event-evidence-{revision['id']}-{mode}",
        )
        description = st.text_area("Description", value=base["description"], key=f"event-description-{revision['id']}-{mode}")
        confidence_levels = ["high", "medium", "low"]
        confidence = st.selectbox(
            "Confidence", confidence_levels, index=confidence_levels.index(base["confidence"]),
            key=f"event-confidence-{revision['id']}-{mode}",
        )
        updated_event = {
            "id": base["id"], "scene_id": scene_id, "character_id": character_id,
            "kind": kind, "timing": timing, "prop_id": prop_id,
            "related_prop_id": related_prop_id, "counterparty_character_id": counterparty_id,
            "fact": fact or None, "value": value or None,
            "relationship_character_id": relationship_character_id,
            "evidence_block_ids": evidence, "description": description, "confidence": confidence,
        }
        if selected_event:
            revised_events = [updated_event if item["id"] == selected_event["id"] else item for item in events]
        else:
            revised_events = [*events, updated_event]
        run_action(
            "Save continuity event",
            lambda: client.patch(
                project["id"], "extraction", "/continuity_events", revised_events, revision["sha256"],
            ),
            success="Continuity event saved and state recalculated.", key=f"save-event-{mode}",
        )
        if selected_event:
            remaining_events = [item for item in events if item["id"] != selected_event["id"]]
            run_action(
                "Delete selected event",
                lambda: client.patch(
                    project["id"], "extraction", "/continuity_events", remaining_events, revision["sha256"],
                ),
                success="Continuity event removed and state recalculated.", key="delete-event",
            )

    with st.expander("Derived before/after state", expanded=not events):
        st.dataframe(payload["state_transitions"], width="stretch")
with tabs[4]:
    st.json(payload["story_contract"])
    if payload.get("extraction_warnings"):
        for warning in payload["extraction_warnings"]:
            st.warning(warning)
with tabs[5]:
    if not issues:
        st.success("No deterministic continuity issues found.")
    transitions_by_key = {
        (item["character_id"], item["scene_id"]): item for item in payload["state_transitions"]
    }
    for issue in issues:
        scene_labels = [scene_names.get(item, item) for item in issue.get("affected_scene_ids", [])]
        entity_label = character_names.get(
            issue.get("entity_id"), element_names.get(issue.get("entity_id"), issue.get("entity_id") or "Project"),
        )
        heading = f"{issue['severity'].upper()} · {issue['code'].replace('_', ' ').title()} · {entity_label}"
        (st.error if issue["severity"] == "blocking" else st.warning)(heading)
        st.write(issue["message"])
        if scene_labels:
            st.markdown("**Affected scenes:** " + " → ".join(scene_labels))
        added = issue.get("added_items", [])
        removed = issue.get("removed_items", [])
        if not added and not removed and issue["code"] in {"prop_transfer_gap", "knowledge_loss"}:
            affected = issue.get("affected_scene_ids", [])
            if len(affected) == 2 and issue.get("entity_id"):
                previous = transitions_by_key.get((issue["entity_id"], affected[0]))
                current = transitions_by_key.get((issue["entity_id"], affected[1]))
                if previous and current:
                    field = "carries" if issue["code"] == "prop_transfer_gap" else "knows"
                    added = sorted(set(current["before"][field]) - set(previous["after"][field]))
                    removed = sorted(set(previous["after"][field]) - set(current["before"][field]))
        if added:
            st.markdown("**Newly present:** " + ", ".join(element_names.get(item, item) for item in added))
        if removed:
            st.markdown("**No longer present:** " + ", ".join(element_names.get(item, item) for item in removed))
        if issue.get("expected") is not None:
            st.markdown(f"**Expected/incoming:** `{issue['expected']}`")
        if issue.get("actual") is not None:
            st.markdown(f"**Actual/next state:** `{issue['actual']}`")
        if issue.get("probable_cause"):
            st.caption("Probable cause: " + issue["probable_cause"])
        if issue.get("suggested_resolution"):
            st.caption("Suggested resolution: " + issue["suggested_resolution"])
        relevant_transitions = [
            transitions_by_key[(issue["entity_id"], scene_id)]
            for scene_id in issue.get("affected_scene_ids", [])
            if (issue.get("entity_id"), scene_id) in transitions_by_key
        ]
        with st.expander("Evidence and technical details"):
            for block_id in issue.get("evidence_block_ids", []):
                if block_id in block_records:
                    st.write(f"Source: {block_records[block_id][1]}")
            if relevant_transitions:
                st.json(relevant_transitions)
            st.json(issue)

if project["stage"] == "extraction_review":
    override = st.text_input("Override reason (required only while blocking issues remain)")
    reviewed = st.checkbox("I reviewed the Issues tab and the extracted canonical records", value=not issues)
    run_action(
        "Approve extraction and continue",
        lambda: client.approve(project["id"], "extraction", revision["id"], override or None),
        success="Extraction approved; cultural grounding and planning completed.", key="approve-extraction",
        disabled=not reviewed or (bool(blocking_issues) and not override.strip()),
    )
