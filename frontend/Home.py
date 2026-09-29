from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client
from frontend.common import setup_page


setup_page("Maidani Mewari Studio")
st.title("Maidani Mewari Screenplay Studio")
st.caption("Structured extraction → cultural plan → approval → adaptation → consistent visual production pack")

try:
    health = client.health()
    st.success(f"Backend connected · AI mode: {health['ai_mode']} · provider: {health['provider']}")
except APIError as error:
    st.error(str(error))
    st.stop()

projects = client.projects()
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
with st.form("create-project"):
    title = st.text_input("Project title", "The Letter")
    source_text = st.text_area("Paste screenplay text", value=default_text, height=360, placeholder="INT. HOUSE - MORNING\n...")
    left, right = st.columns(2)
    with left:
        culture = st.selectbox("Culture / dialect", ["Maidani Mewari"])
        locality = st.text_input("Exact locality", "Rajsamand plains, Rajasthan")
    with right:
        setting = st.selectbox("Setting", ["rural", "urban"])
        output_script = st.selectbox("Output writing script", ["Devanagari"])
    submitted = st.form_submit_button("Create project", type="primary")
if submitted:
    try:
        project = client.create({
            "title": title, "source_text": source_text, "culture": culture, "locality": locality,
            "setting": setting, "period": "Contemporary 2020-2026", "output_script": output_script,
        })
        st.session_state.project_id = project["id"]
        st.success("Project created. Open Scene Extraction from the sidebar.")
    except APIError as error:
        st.error(str(error))

