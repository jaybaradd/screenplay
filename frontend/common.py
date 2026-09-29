from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client


STAGES = [
    "created", "extracting", "extraction_review", "researching_culture", "plan_review",
    "adapting", "screenplay_review", "building_visual_manifest", "visual_review",
    "preparing_character_assets", "character_images_review", "preparing_scene_assets", "visuals_ready",
]


def setup_page(title: str, icon: str = "🎬") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown("""<style>
    :root {
        --studio-text: #3d2b22;
        --studio-muted: #6f594d;
        --studio-background: #fbf7ef;
        --studio-surface: #fffdf8;
        --studio-sidebar: #f0e6d7;
        --studio-border: #d8c9b8;
        --studio-accent: #7f3f2f;
    }
    html, body, .stApp {
        background: var(--studio-background);
        color: var(--studio-text);
    }
    .stApp p, .stApp li, .stApp label, .stApp small,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--studio-text);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #35261d !important;
    }
    a { color: #7a3528 !important; }
    [data-testid="stSidebar"] {
        background: var(--studio-sidebar);
        border-right: 1px solid var(--studio-border);
    }
    [data-testid="stSidebar"] * { color: var(--studio-text); }
    [data-testid="stMetric"] {
        background: var(--studio-surface);
        border: 1px solid var(--studio-border);
        padding: 12px;
        border-radius: 10px;
    }
    input, textarea, [data-baseweb="select"] > div {
        background: var(--studio-surface) !important;
        color: var(--studio-text) !important;
        -webkit-text-fill-color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    [data-baseweb="tab-list"] button,
    [data-testid="stExpander"] summary,
    [data-testid="stFileUploader"] {
        color: var(--studio-text) !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: var(--studio-surface);
        border-color: var(--studio-border);
        color: var(--studio-text);
    }
    [data-testid="stBaseButton-secondary"] p { color: var(--studio-text) !important; }
    [data-testid="stBaseButton-primary"] {
        background: var(--studio-accent);
        border-color: var(--studio-accent);
    }
    [data-testid="stBaseButton-primary"] p { color: white !important; }
    .stage { padding: 8px 12px; border-radius: 999px; background: #7f3f2f; color: white; display: inline-block; }
    .stage * { color: white !important; }
    </style>""", unsafe_allow_html=True)


def require_project() -> dict:
    project_id = st.session_state.get("project_id")
    if not project_id:
        st.warning("Choose or create a project on the Home page first.")
        st.stop()
    try:
        project = client.project(project_id)
    except APIError as error:
        st.error(str(error))
        st.stop()
    st.sidebar.markdown(f"**{project['title']}**")
    st.sidebar.markdown(f"<span class='stage'>{project['stage'].replace('_',' ')}</span>", unsafe_allow_html=True)
    st.sidebar.caption(f"{project['culture']} · {project['locality']} · {project['setting']}")
    return project


def run_action(label: str, action, *, success: str = "Saved", key: str | None = None, disabled: bool = False):
    if st.button(label, key=key, disabled=disabled, type="primary"):
        try:
            result = action()
            st.success(success)
            st.rerun()
        except APIError as error:
            st.error(str(error))
