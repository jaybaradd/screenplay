from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client
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

tabs = st.tabs(["Scenes", "Characters", "Production", "State", "Story contract", "Issues"])
with tabs[0]:
    edited_scenes = st.data_editor(payload["scenes"], use_container_width=True, num_rows="fixed", key=f"scenes-{revision['id']}")
    run_action("Save scene edits", lambda: client.patch(project["id"], "extraction", "/scenes", edited_scenes, revision["sha256"]), key="save-scenes")
with tabs[1]:
    edited_characters = st.data_editor(payload["characters"], use_container_width=True, num_rows="fixed", key=f"characters-{revision['id']}")
    run_action("Save character edits", lambda: client.patch(project["id"], "extraction", "/characters", edited_characters, revision["sha256"]), key="save-characters")
    if len(payload["characters"]) > 1:
        labels = {f"{item['name']} · {item['id'][:8]}": item["id"] for item in payload["characters"]}
        primary_label = st.selectbox("Canonical character", labels, key="merge-primary")
        duplicates = st.multiselect("Aliases / duplicate records to merge", [label for label in labels if label != primary_label])
        run_action("Merge selected records", lambda: client.merge(project["id"], "character", labels[primary_label], [labels[item] for item in duplicates]), key="merge", disabled=not duplicates)
with tabs[2]:
    edited_elements = st.data_editor(payload["production_elements"], use_container_width=True, num_rows="fixed", key=f"elements-{revision['id']}")
    run_action("Save production edits", lambda: client.patch(project["id"], "extraction", "/production_elements", edited_elements, revision["sha256"]), key="save-production")
with tabs[3]:
    st.dataframe(payload["state_transitions"], use_container_width=True)
with tabs[4]:
    st.json(payload["story_contract"])
    if payload.get("extraction_warnings"):
        for warning in payload["extraction_warnings"]:
            st.warning(warning)
with tabs[5]:
    issues = project["revisions"].get("continuity_issues", {}).get("payload", [])
    if not issues:
        st.success("No deterministic continuity issues found.")
    for issue in issues:
        (st.error if issue["severity"] == "blocking" else st.warning)(f"{issue['code']}: {issue['message']} · scenes {issue.get('affected_scene_ids', [])}")

if project["stage"] == "extraction_review":
    override = st.text_input("Override reason (required only while blocking issues remain)")
    run_action("Approve extraction and continue", lambda: client.approve(project["id"], "extraction", revision["id"], override or None), success="Extraction approved; cultural grounding and planning completed.", key="approve-extraction")
