from __future__ import annotations

import streamlit as st

from frontend.client import canonical_hash, client
from frontend.common import require_project, run_action, setup_page


setup_page("Screenplay Comparison", "📝")
project = require_project()
st.title("Source and adapted screenplay")
source_revision = project["revisions"].get("extraction")
adapted_revision = project["revisions"].get("adapted_screenplay")
if not source_revision or not adapted_revision:
    st.info("Approve the adaptation plan first.")
    st.stop()

source_scenes = source_revision["payload"]["scenes"]
adapted_scenes = adapted_revision["payload"]["scenes"]
block_options = {}
for source_scene, adapted_scene in zip(source_scenes, adapted_scenes):
    st.subheader(f"Scene {source_scene['number']} · {source_scene['heading']}")
    left, right = st.columns(2)
    with left:
        st.caption("SOURCE")
        for block in source_scene["blocks"]:
            if block.get("speaker_label"):
                st.markdown(f"**{block['speaker_label']}**")
            st.write(block["text"])
    with right:
        st.caption("ADAPTED")
        for block in adapted_scene["blocks"]:
            label = f"Scene {source_scene['number']} · {block['id'][:8]} · {block['adapted_text'][:55]}"
            block_options[label] = block
            st.write(block["adapted_text"])
            with st.expander("Why this changed"):
                st.write(block["explanation"])
                st.caption(f"Confidence: {block['confidence']} · dimensions: {', '.join(block['changed_dimensions']) or 'none'}")
    st.divider()

st.subheader("Surgical correction")
selected_label = st.selectbox("Target exactly one block", list(block_options))
selected = block_options[selected_label]
instruction = st.text_area("Replacement instruction", placeholder="In mock mode, enter the exact replacement text. In live mode, describe the desired correction.")
run_action(
    "Apply only this correction",
    lambda: client.correct(project["id"], selected["id"], instruction, canonical_hash(selected)),
    success="A new screenplay revision was created; unrelated blocks are unchanged.",
    key="correct", disabled=not instruction.strip(),
)

verification = project["revisions"].get("adaptation_verification", {}).get("payload", [])
for issue in verification:
    st.error(issue["message"])
if project["stage"] == "screenplay_review":
    override = st.text_input("Override reason (only for unresolved blocking verification issues)")
    run_action("Approve screenplay and build visual specs", lambda: client.approve(project["id"], "screenplay", adapted_revision["id"], override or None), success="Screenplay approved; visual manifest created.", key="approve-screenplay")

