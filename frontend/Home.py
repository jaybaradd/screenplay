from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client
from frontend.common import setup_page


setup_page("Cultural Adaptation Studio")
st.title("Cultural Screenplay Adaptation Studio")
st.caption("Structured extraction → cultural plan → approval → adaptation → consistent visual production pack")

try:
    health = client.health()
    st.success(f"Backend connected · AI mode: {health['ai_mode']} · provider: {health['provider']}")
except APIError as error:
    st.error(str(error))
    st.stop()

projects = client.projects()
cultures = client.cultures()
if projects:
    options = {f"{item['title']} — {item['stage']}": item["id"] for item in projects}
    selected = st.selectbox("Reopen a project", ["Create a new project", *options])
    if selected != "Create a new project":
        st.session_state.project_id = options[selected]
        st.info("Project selected. Use the pages in the sidebar to continue.")

st.divider()
st.subheader("Start a new adaptation")
uploaded = st.file_uploader("Optional .txt screenplay", type=["txt"])
default_text = uploaded.getvalue().decode("utf-8") if uploaded else ""
if not cultures:
    st.error("No production culture profiles are installed.")
    st.stop()
profile_labels = {item["display_name"]: item for item in cultures}
with st.form("create-project"):
    title = st.text_input("Project title")
    source_text = st.text_area("Paste screenplay text", value=default_text, height=360)
    left, right = st.columns(2)
    with left:
        culture_label = st.selectbox("Culture / dialect", list(profile_labels))
        profile = profile_labels[culture_label]
        st.caption(f"Target variety: {profile['target_variety']}")
        st.caption(profile["geographic_scope"])
        if profile.get("limitations"):
            with st.expander("Profile limitations", expanded=False):
                for limitation in profile["limitations"]:
                    st.write(f"• {limitation}")
        locality = st.text_input("Exact locality")
    with right:
        setting = st.selectbox("Setting", profile["supported_settings"])
        period_labels = {item["display_name"]: item["id"] for item in profile["supported_periods"]}
        period_label = st.selectbox("Period", list(period_labels))
        script_labels = {item["display_name"]: item["id"] for item in profile["supported_scripts"]}
        script_label = st.selectbox("Output writing script", list(script_labels))
    submitted = st.form_submit_button("Create project", type="primary")
if submitted:
    try:
        project = client.create({
            "title": title, "source_text": source_text, "culture_id": profile["culture_id"], "locality": locality,
            "setting": setting, "period": period_labels[period_label], "output_script": script_labels[script_label],
        })
        st.session_state.project_id = project["id"]
        st.success("Project created. Open Scene Extraction from the sidebar.")
    except APIError as error:
        st.error(str(error))
