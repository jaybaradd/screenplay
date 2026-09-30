from __future__ import annotations

import streamlit as st

from frontend.client import client
from frontend.common import require_project, run_action, setup_page


setup_page("Adaptation Plan", "🧭")
project = require_project()
st.title("Grounded cultural brief and six-layer plan")
brief = project["revisions"].get("cultural_brief")
plan = project["revisions"].get("adaptation_plan")
if not brief or not plan:
    st.info("Approve extraction first. The graph will build the cultural brief and adaptation plan.")
    st.stop()

st.subheader("Frozen cultural brief")
st.caption(f"{brief['payload']['culture']} · {brief['payload']['locality']} · {brief['payload']['setting']} · {brief['payload']['period']}")
st.write(brief["payload"]["research_summary"])
st.dataframe(brief["payload"]["claims"], width='stretch')
guide = brief["payload"].get("dialect_guide")
if guide:
    st.subheader("Approved language guide")
    st.caption(f"{guide['target_variety']} · {guide['writing_script']}")
    if guide.get("features"):
        st.dataframe(guide["features"], width="stretch")
    else:
        st.error("No source-backed dialect features were found. Do not approve a Mewari adaptation yet.")
    for rule in guide.get("code_switching_rules", []):
        st.info("Code-switching: " + rule)
else:
    st.error("The cultural brief has no Maidani Mewari dialect guide.")
for constraint in brief["payload"]["negative_constraints"]:
    st.warning(constraint)

verification = project["revisions"].get("plan_verification", {}).get("payload", [])
if verification:
    st.subheader("Plan quality checks")
    for issue in verification:
        (st.error if issue["severity"] == "blocking" else st.warning)(f"{issue['code']}: {issue['message']}")

st.subheader("Adaptation layers")
layer_tabs = st.tabs([item["layer"].replace("_", " ").title() for item in plan["payload"]["layers"]])
for tab, layer in zip(layer_tabs, plan["payload"]["layers"]):
    with tab:
        st.write(layer["objective"])
        for decision in layer["decisions"]:
            st.markdown(f"**{decision['id']}**")
            st.write("Proposed changes", decision["proposed_changes"])
            st.write("Preserved invariants", decision["preserved_invariants"])
            if decision["uncertainty"]:
                st.warning(" · ".join(decision["uncertainty"]))
            st.caption("Negative constraints: " + " · ".join(decision["negative_constraints"]))

st.subheader("Synthesis")
for item in plan["payload"]["synthesis"]:
    st.write("•", item)
if plan["payload"]["deeper_change_requests"]:
    st.error("Deeper changes require explicit approval: " + " · ".join(plan["payload"]["deeper_change_requests"]))

if project["stage"] == "plan_review":
    override = st.text_input("Override reason (only if you intentionally accept blocking plan issues)")
    run_action(
        "Approve plan and adapt screenplay",
        lambda: client.approve(project["id"], "plan", plan["id"], override or None),
        success="Plan approved and screenplay adapted.", key="approve-plan",
    )
