from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client
from frontend.common import require_project, run_action, setup_page


setup_page("Visual Gallery", "🎨")
project = require_project()
st.title("Character, costume and scene visual pack")
manifest = project["revisions"].get("visual_manifest")
if not manifest:
    st.info("Approve the adapted screenplay first.")
    st.stop()

with st.expander("Approved visual prompts", expanded=project["stage"] == "visual_review"):
    st.subheader("Character and costume appearances")
    for appearance in manifest["payload"]["appearances"]:
        st.markdown(f"**{appearance['id'][:8]} · character {appearance['character_id'][:8]}**")
        st.write(appearance["prompt"])
    st.subheader("Scene prompts")
    for scene in manifest["payload"]["scenes"]:
        st.markdown(f"**{scene['id'][:8]} · scene {scene['scene_id'][:8]}**")
        st.write(scene["prompt"])
        st.caption("Avoid: " + scene["negative_prompt"])

if project["stage"] == "visual_review":
    run_action("Approve prompts and prepare character sheets", lambda: client.approve(project["id"], "visuals", manifest["id"]), success="Character-sheet jobs prepared.", key="approve-visuals")

assets = project["assets"]
verification_records = project["revisions"].get("visual_verification", {}).get("payload", [])
latest_verification = {item["asset_id"]: item for item in verification_records}
if assets:
    for kind, heading in (("character_sheet", "Character & costume bible"), ("scene_keyframe", "Scene keyframes")):
        selected = [asset for asset in assets if asset["kind"] == kind]
        if not selected:
            continue
        st.subheader(heading)
        columns = st.columns(3)
        for index, asset in enumerate(selected):
            with columns[index % 3]:
                st.caption(f"{asset['canonical_id'][:8]} · {asset['status']} · attempt {asset['attempt']}")
                if asset["path"]:
                    try:
                        st.image(client.asset_bytes(asset["id"]), width='stretch')
                    except APIError:
                        st.warning("Image file unavailable")
                verification = latest_verification.get(asset["id"])
                if verification:
                    if verification["passed"]:
                        st.success("Visual verification passed")
                    else:
                        st.error(verification["summary"])
                        for issue in verification["issues"]:
                            st.caption(f"{issue['dimension']}: {issue['message']}")
                if asset["status"] in {"pending", "failed", "invalidated"}:
                    run_action("Generate" if asset["status"] == "pending" else "Retry only this", lambda asset_id=asset["id"]: client.generate_asset(asset_id, retry=asset["status"] != "pending"), key=f"gen-{asset['id']}")
                if kind == "character_sheet" and asset["status"] == "generated":
                    run_action("Approve reference", lambda asset_id=asset["id"]: client.approve_asset(asset_id), key=f"approve-{asset['id']}")

if project["stage"] == "character_images_review":
    sheets = [asset for asset in assets if asset["kind"] == "character_sheet"]
    ready = bool(sheets) and all(asset["status"] == "approved" for asset in sheets)
    run_action("Lock character references and prepare keyframes", lambda: client.approve(project["id"], "character_images"), success="Scene-keyframe jobs prepared.", key="approve-character-images", disabled=not ready)
