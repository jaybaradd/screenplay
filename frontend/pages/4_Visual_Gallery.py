from __future__ import annotations

import copy

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

editable_manifest = copy.deepcopy(manifest["payload"])
with st.expander("Visual prompts", expanded=project["stage"] == "visual_review"):
    st.subheader("Character and costume appearances")
    for index, appearance in enumerate(editable_manifest["appearances"]):
        st.markdown(f"**{appearance['id'][:8]} · character {appearance['character_id'][:8]}**")
        if project["stage"] == "visual_review":
            appearance["identity_description"] = st.text_area(
                "Identity description", appearance["identity_description"],
                key=f"appearance-identity-{manifest['id']}-{appearance['id']}",
            )
            appearance["costume_description"] = st.text_area(
                "Costume description", appearance["costume_description"],
                key=f"appearance-costume-{manifest['id']}-{appearance['id']}",
            )
            appearance["grooming_description"] = st.text_area(
                "Grooming description", appearance["grooming_description"],
                key=f"appearance-grooming-{manifest['id']}-{appearance['id']}",
            )
            appearance["prompt"] = st.text_area(
                "Final image-generation prompt", appearance["prompt"], height=220,
                key=f"appearance-prompt-{manifest['id']}-{appearance['id']}",
            )
        else:
            st.write(appearance["identity_description"])
            st.write(appearance["costume_description"])
            st.write(appearance["grooming_description"])
            st.write(appearance["prompt"])
        if index < len(editable_manifest["appearances"]) - 1:
            st.divider()
    st.subheader("Scene prompts")
    for scene in editable_manifest["scenes"]:
        st.markdown(f"**{scene['id'][:8]} · scene {scene['scene_id'][:8]}**")
        if project["stage"] == "visual_review":
            scene["prompt"] = st.text_area(
                "Final scene image-generation prompt", scene["prompt"], height=220,
                key=f"scene-prompt-{manifest['id']}-{scene['id']}",
            )
            scene["negative_prompt"] = st.text_area(
                "Visual negative constraints", scene["negative_prompt"],
                key=f"scene-negative-{manifest['id']}-{scene['id']}",
            )
        else:
            st.write(scene["prompt"])
            st.caption("Avoid: " + scene["negative_prompt"])

if project["stage"] == "visual_review":
    has_unsaved_changes = editable_manifest != manifest["payload"]
    st.warning("The currently displayed revision is editable. Save changes before approving it.")
    left, middle, right = st.columns(3)
    with left:
        run_action(
            "Regenerate detailed visual manifest",
            lambda: client.regenerate_visuals(project["id"]),
            success="A new detailed visual-manifest revision was created.", key="regenerate-visuals",
        )
    with middle:
        run_action(
            "Save all prompt edits",
            lambda: client.patch(
                project["id"], "visual_manifest", "/", editable_manifest, manifest["sha256"],
            ),
            success="Prompt edits saved as a new visual-manifest revision.", key="save-visual-prompts",
            disabled=not has_unsaved_changes,
        )
    with right:
        run_action(
            "Approve prompts and prepare character sheets",
            lambda: client.approve(project["id"], "visuals", manifest["id"]),
            success="Character-sheet jobs prepared.", key="approve-visuals", disabled=has_unsaved_changes,
        )

assets = project["assets"]
verification_records = project["revisions"].get("visual_verification", {}).get("payload", [])
latest_verification = {item["asset_id"]: item for item in verification_records}
characters = {
    item["id"]: item["name"]
    for item in project["revisions"].get("extraction", {}).get("payload", {}).get("characters", [])
}
appearance_labels = {
    item["id"]: characters.get(item["character_id"], item["character_id"][:8])
    for item in manifest["payload"]["appearances"]
}
scene_numbers = {
    item["id"]: item["number"]
    for item in project["revisions"].get("extraction", {}).get("payload", {}).get("scenes", [])
}
scene_labels = {
    item["id"]: f"Scene {scene_numbers.get(item['scene_id'], item['scene_id'][:8])}"
    for item in manifest["payload"]["scenes"]
}


def split_notes(value: str) -> list[str]:
    return [item.strip() for line in value.splitlines() for item in line.split(",") if item.strip()]


def show_verification(asset: dict) -> None:
    verification = latest_verification.get(asset["id"])
    if not verification:
        return
    if verification["passed"]:
        st.success("Visual verification passed")
    else:
        st.error(verification["summary"])
        for issue in verification["issues"]:
            st.caption(f"{issue['dimension']}: {issue['message']}")


def show_correction(asset: dict, kind: str) -> None:
    allowed = (
        project["stage"] == "character_images_review" if kind == "character_sheet"
        else project["stage"] in {"scene_images_review", "visuals_ready"}
    )
    if not allowed or asset["status"] not in {"generated", "failed", "approved"} or not asset["path"]:
        return
    with st.expander("Correct only this image"):
        feedback = st.text_area(
            "Describe only what should change",
            placeholder="Keep every approved identity, costume and set detail unchanged; correct only …",
            key=f"asset-feedback-{asset['id']}",
        )
        candidates = [
            item for item in assets if item["kind"] == kind and item["id"] != asset["id"]
            and item["status"] in {"generated", "approved"} and item["path"]
        ]
        options = {"No additional style reference": None}
        for reference in candidates:
            label = (
                appearance_labels.get(reference["canonical_id"], reference["canonical_id"][:8])
                if kind == "character_sheet" else scene_labels.get(reference["canonical_id"], reference["canonical_id"][:8])
            )
            options[f"Style reference: {label}"] = reference["id"]
        selected_reference = st.selectbox(
            "Optional style reference", list(options), key=f"asset-style-reference-{asset['id']}",
        )
        run_action(
            "Apply feedback and regenerate",
            lambda asset_id=asset["id"], instruction=feedback, reference_id=options[selected_reference]:
                client.correct_asset(asset_id, instruction, reference_id),
            success="A corrected version was generated; the previous version remains in history.",
            key=f"correct-asset-{asset['id']}", disabled=len(feedback.strip()) < 3,
        )


character_assets = [
    asset for asset in assets if asset["kind"] == "character_sheet" and asset["status"] != "invalidated"
]
if character_assets:
    st.subheader("Character & costume bible")
    columns = st.columns(3)
    for index, asset in enumerate(character_assets):
        with columns[index % 3]:
            st.markdown(f"**{appearance_labels.get(asset['canonical_id'], asset['canonical_id'][:8])}**")
            st.caption(f"{asset['status']} · generation attempt {asset['attempt']}")
            if asset["path"]:
                try:
                    st.image(client.asset_bytes(asset["id"]), width="stretch")
                except APIError:
                    st.warning("Image file unavailable")
            show_verification(asset)
            if asset["status"] in {"pending", "failed"}:
                run_action(
                    "Generate" if asset["status"] == "pending" else "Retry same prompt",
                    lambda asset_id=asset["id"], retry=asset["status"] != "pending": client.generate_asset(asset_id, retry=retry),
                    key=f"gen-{asset['id']}",
                )
            if asset["status"] == "generated":
                run_action("Approve reference", lambda asset_id=asset["id"]: client.approve_asset(asset_id), key=f"approve-{asset['id']}")
            show_correction(asset, "character_sheet")

continuity_revision = project["revisions"].get("visual_continuity")
if project["stage"] == "visuals_ready" and not continuity_revision:
    st.warning(
        "This project predates sequential scene continuity. Enabling it preserves every existing image and lets you "
        "approve Scenes 1 and 2 as references before rebuilding Scene 3."
    )
    run_action(
        "Enable sequential visual continuity",
        lambda: client.start_visual_continuity(project["id"]),
        success="Sequential continuity enabled. Existing images were preserved.", key="start-visual-continuity",
    )

ledger = (continuity_revision or {}).get("payload", {"sets": [], "scenes": []})
approved_scenes = {item["scene_id"]: item for item in ledger.get("scenes", [])}
set_snapshots = {item["set_id"]: item for item in ledger.get("sets", [])}
compiled_asset_ids = {
    item["asset_id"]
    for item in project["revisions"].get("scene_prompt_compilations", {}).get("payload", [])
}
compilation_by_asset = {
    item["asset_id"]: item
    for item in project["revisions"].get("scene_prompt_compilations", {}).get("payload", [])
}
scene_source = project["revisions"].get("extraction", {}).get("payload", {}).get("scenes", [])
source_by_id = {item["id"]: item for item in scene_source}
active_scene_assets = {
    asset["scene_id"]: asset for asset in assets
    if asset["kind"] == "scene_keyframe" and asset["status"] != "invalidated"
}

if any(asset["kind"] == "scene_keyframe" for asset in assets) or project["stage"] == "scene_images_review":
    st.subheader("Sequential scene keyframes")
    st.caption(
        "Approve scenes in order. The most recent approved image of the same exact set controls architecture; "
        "the preceding scene carries only character, costume and prop state."
    )
    for index, spec in enumerate(manifest["payload"]["scenes"]):
        source_scene = source_by_id.get(spec["scene_id"], {})
        sub_location = spec.get("sub_location") or source_scene.get("sub_location") or source_scene.get("location", "Unknown set")
        asset = active_scene_assets.get(spec["scene_id"])
        prior_ready = all(item["scene_id"] in approved_scenes for item in manifest["payload"]["scenes"][:index])
        existing_set = set_snapshots.get(spec.get("set_id", "")) or next((
            item for item in ledger.get("sets", [])
            if item.get("name", "").strip().casefold() == sub_location.strip().casefold()
        ), None)
        requires_set_recompile = bool(
            asset and existing_set and existing_set.get("source_scene_id") != spec["scene_id"]
            and asset["id"] not in compiled_asset_ids
        )
        with st.container(border=True):
            st.markdown(f"### Scene {source_scene.get('number', index + 1)} · {sub_location}")
            st.caption(
                f"set {str(spec.get('set_id') or 'derived on compile')[:8]} · "
                + (f"{asset['status']} · attempt {asset['attempt']}" if asset else "locked / not compiled")
            )
            if asset:
                with st.expander("Generation prompt and reference bundle"):
                    can_edit_prompt = (
                        project["stage"] == "scene_images_review"
                        and asset["status"] in {"pending", "generated", "failed"}
                    )
                    edited_prompt = st.text_area(
                        "Compiled prompt", asset["prompt"], height=260, disabled=not can_edit_prompt,
                        key=f"compiled-prompt-{asset['id']}",
                    )
                    if can_edit_prompt:
                        run_action(
                            "Save as a new prompt version",
                            lambda asset_id=asset["id"], prompt=edited_prompt, dependency=asset["dependency_hash"]:
                                client.revise_asset_prompt(asset_id, prompt, dependency),
                            success="A new pending prompt version was saved. No image was generated yet.",
                            key=f"save-asset-prompt-{asset['id']}",
                            disabled=edited_prompt.strip() == asset["prompt"].strip(),
                        )
                    compilation = compilation_by_asset.get(asset["id"])
                    if compilation and compilation.get("reference_assets"):
                        st.markdown("**Labelled references, in priority order**")
                        for reference in compilation["reference_assets"]:
                            st.caption(reference["label"])
            if asset and asset["path"]:
                try:
                    st.image(client.asset_bytes(asset["id"]), width="stretch")
                except APIError:
                    st.warning("Image file unavailable")
                show_verification(asset)
            if not prior_ready:
                st.info("Locked until the preceding scene is approved.")
                continue
            if not asset:
                run_action(
                    "Compile this scene from approved continuity",
                    lambda scene_id=spec["scene_id"]: client.compile_scene(project["id"], scene_id),
                    success="The scene prompt was compiled from approved references.", key=f"compile-{spec['scene_id']}",
                )
                continue
            if asset["status"] in {"pending", "failed"}:
                run_action(
                    "Generate" if asset["status"] == "pending" else "Retry same prompt",
                    lambda asset_id=asset["id"], retry=asset["status"] != "pending": client.generate_asset(asset_id, retry=retry),
                    key=f"gen-{asset['id']}",
                )
            if project["stage"] == "scene_images_review" and asset["status"] in {"generated", "failed"} and index > 0:
                if requires_set_recompile:
                    st.warning(
                        "This set already has an approved visual authority. Recompile before approval so this image receives "
                        "that same-set reference."
                    )
                run_action(
                    "Recompile with approved continuity",
                    lambda scene_id=spec["scene_id"]: client.compile_scene(project["id"], scene_id),
                    success="A new pending version now uses the approved same-set and prior-scene references.",
                    key=f"recompile-{asset['id']}",
                )
            override_reference_gate = False
            reference_override_reason = None
            if requires_set_recompile and asset["path"]:
                with st.expander("Keep this existing image without recompiling"):
                    st.caption(
                        "Use this only when you have visually confirmed that the existing image already matches the approved "
                        "set geometry and character references. The decision is recorded in the audit trail."
                    )
                    override_reference_gate = st.checkbox(
                        "I confirm this image is continuity-consistent",
                        key=f"override-reference-{asset['id']}",
                    )
                    reference_override_reason = st.text_area(
                        "Continuity override reason",
                        placeholder="The dining-room geometry, character appearances and costumes match the approved references.",
                        key=f"reference-override-reason-{asset['id']}",
                    )
            can_approve_image = (
                asset["status"] == "generated"
                or (asset["status"] == "failed" and bool(asset["path"]))
            )
            reference_gate_satisfied = (
                not requires_set_recompile
                or (override_reference_gate and len((reference_override_reason or "").strip()) >= 10)
            )
            if can_approve_image and project["stage"] == "scene_images_review" and reference_gate_satisfied:
                with st.expander("Approve image and record continuity", expanded=True):
                    st.caption("Record only details visible in this approved image. These become constraints, not cultural claims.")
                    geometry = st.text_area(
                        "Set geometry (door/window count and positions, room proportions)",
                        key=f"geometry-{asset['id']}",
                    )
                    fixed = st.text_area("Fixed furniture and architectural elements (one per line)", key=f"fixed-{asset['id']}")
                    materials = st.text_input("Materials (comma-separated)", key=f"materials-{asset['id']}")
                    palette = st.text_input("Stable colour palette (comma-separated)", key=f"palette-{asset['id']}")
                    adjacency = st.text_area(
                        "Explicit adjacency only (example: east door opens to the veranda)", key=f"adjacency-{asset['id']}",
                    )
                    mutable = st.text_input("Elements allowed to move/change", key=f"mutable-{asset['id']}")
                    character_state = st.text_area("Character/costume state carried forward", key=f"character-state-{asset['id']}")
                    prop_state = st.text_area("Prop state carried forward", key=f"prop-state-{asset['id']}")
                    promote = st.checkbox(
                        "Use this image as the authoritative reference for this exact set",
                        value=existing_set is None, key=f"promote-set-{asset['id']}",
                        help="Leave off for a later scene unless it intentionally corrects the canonical set.",
                    )
                    verification = latest_verification.get(asset["id"])
                    verification_failed = asset["status"] == "failed" or bool(
                        verification and not verification.get("passed")
                    )
                    override_verification = False
                    override_reason = None
                    if verification_failed:
                        st.warning(
                            "The automated verifier rejected this image or failed to return usable structured output. You may "
                            "still approve it when the visible result is acceptable; the override and reason will be preserved "
                            "in the audit log."
                        )
                        override_verification = st.checkbox(
                            "I reviewed the image and want to override this verification failure",
                            key=f"override-verification-{asset['id']}",
                        )
                        override_reason = st.text_area(
                            "Override reason",
                            placeholder=(
                                "Example: The intended slight hand-to-palm overlap is visually acceptable; the verifier "
                                "overweighted a minute contact detail while identity, costume and set continuity are correct."
                            ),
                            key=f"override-reason-{asset['id']}",
                        )
                    payload = {
                        "geometry_notes": geometry, "fixed_elements": split_notes(fixed),
                        "materials": split_notes(materials), "palette": split_notes(palette),
                        "adjacency_notes": split_notes(adjacency), "mutable_elements": split_notes(mutable),
                        "character_state_notes": split_notes(character_state), "prop_state_notes": split_notes(prop_state),
                        "promote_as_set_reference": promote,
                        "override_verification": override_verification,
                        "override_reason": override_reason,
                        "override_reference_gate": override_reference_gate,
                        "reference_override_reason": reference_override_reason,
                    }
                    run_action(
                        "Approve scene and unlock the next",
                        lambda asset_id=asset["id"], data=payload: client.approve_scene_asset(asset_id, data),
                        success="Scene continuity was frozen and the next scene was unlocked.", key=f"approve-scene-{asset['id']}",
                        disabled=(verification_failed and (
                            not override_verification or len((override_reason or "").strip()) < 10
                        )) or not reference_gate_satisfied,
                    )
            elif spec["scene_id"] in approved_scenes:
                snapshot = approved_scenes[spec["scene_id"]]
                st.success("Approved as a continuity checkpoint")
                st.caption(
                    f"Characters: {len(snapshot.get('appearance_ids', []))} · props: {len(snapshot.get('prop_ids', []))}"
                )
            show_correction(asset, "scene_keyframe")

invalidated_scenes = [asset for asset in assets if asset["kind"] == "scene_keyframe" and asset["status"] == "invalidated"]
if invalidated_scenes:
    with st.expander(f"Previous scene versions ({len(invalidated_scenes)})"):
        for asset in invalidated_scenes:
            left, right = st.columns([1, 2])
            with left:
                if asset["path"]:
                    try:
                        st.image(client.asset_bytes(asset["id"]), width="stretch")
                    except APIError:
                        st.caption("Image file unavailable")
            with right:
                st.markdown(f"**{scene_labels.get(asset['canonical_id'], asset['canonical_id'][:8])}**")
                st.caption(f"version {asset['id'][:8]} · retained for audit/history")
                verification = latest_verification.get(asset["id"])
                if verification:
                    st.caption(verification.get("summary", ""))
                if asset["path"] and project["stage"] in {"scene_images_review", "visuals_ready"}:
                    run_action(
                        "Restore this image without regenerating",
                        lambda asset_id=asset["id"]: client.restore_asset(asset_id),
                        success="This retained image is active again. No generation call was made.",
                        key=f"restore-asset-{asset['id']}",
                    )
            st.divider()

if project["stage"] == "character_images_review":
    sheets = [
        asset for asset in assets
        if asset["kind"] == "character_sheet" and asset["status"] != "invalidated"
    ]
    ready = bool(sheets) and all(asset["status"] == "approved" for asset in sheets)
    run_action("Lock character references and prepare keyframes", lambda: client.approve(project["id"], "character_images"), success="Scene-keyframe jobs prepared.", key="approve-character-images", disabled=not ready)
