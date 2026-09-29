from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client
from frontend.common import require_project, setup_page


setup_page("Continuity & Export", "📦")
project = require_project()
st.title("Continuity, traceability and export")

issues = project["revisions"].get("continuity_issues", {}).get("payload", [])
blocking = [item for item in issues if item.get("severity") == "blocking"]
left, middle, right = st.columns(3)
left.metric("Continuity issues", len(issues))
middle.metric("Blocking", len(blocking))
right.metric("Assets", len(project["assets"]))
if not issues:
    st.success("No deterministic continuity issues recorded.")
for issue in issues:
    with st.expander(f"{issue['severity'].upper()} · {issue['code']}"):
        st.write(issue["message"])
        st.write("Affected scenes", issue.get("affected_scene_ids", []))
        if issue.get("suggested_resolution"):
            st.info(issue["suggested_resolution"])

st.subheader("Traceability")
adapted = project["revisions"].get("adapted_screenplay", {}).get("payload")
if adapted:
    rows = []
    for scene in adapted["scenes"]:
        for block in scene["blocks"]:
            rows.append({
                "scene": scene["source_scene_id"][:8], "adapted_block": block["id"][:8],
                "source_blocks": ", ".join(item[:8] for item in block["source_block_ids"]),
                "layers": ", ".join(block["adaptation_layer_ids"]), "confidence": block["confidence"],
            })
    st.dataframe(rows, width='stretch')

st.subheader("Download output package")
if adapted:
    try:
        package = client.export(project["id"])
        st.download_button("Download project-export.zip", package, file_name="project-export.zip", mime="application/zip", type="primary")
    except APIError as error:
        st.error(str(error))
else:
    st.info("Adapt the screenplay before exporting.")

