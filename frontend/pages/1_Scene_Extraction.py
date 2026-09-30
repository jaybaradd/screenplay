from __future__ import annotations

import copy

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
    character_names = {item["id"]: item["name"] for item in payload["characters"]}
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
    st.dataframe(payload["state_transitions"], width="stretch")
with tabs[4]:
    st.json(payload["story_contract"])
    if payload.get("extraction_warnings"):
        for warning in payload["extraction_warnings"]:
            st.warning(warning)
with tabs[5]:
    if not issues:
        st.success("No deterministic continuity issues found.")
    for issue in issues:
        (st.error if issue["severity"] == "blocking" else st.warning)(
            f"{issue['code']}: {issue['message']} · scenes {issue.get('affected_scene_ids', [])}"
        )
        if issue.get("suggested_resolution"):
            st.caption("Suggested resolution: " + issue["suggested_resolution"])

if project["stage"] == "extraction_review":
    override = st.text_input("Override reason (required only while blocking issues remain)")
    reviewed = st.checkbox("I reviewed the Issues tab and the extracted canonical records", value=not issues)
    run_action(
        "Approve extraction and continue",
        lambda: client.approve(project["id"], "extraction", revision["id"], override or None),
        success="Extraction approved; cultural grounding and planning completed.", key="approve-extraction",
        disabled=not reviewed or (bool(blocking_issues) and not override.strip()),
    )
