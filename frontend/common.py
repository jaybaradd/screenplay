from __future__ import annotations

import streamlit as st

from frontend.client import APIError, client


STAGES = [
    "created", "extracting", "extraction_review", "researching_culture", "plan_review",
    "adapting", "screenplay_review", "building_visual_manifest", "visual_review",
    "preparing_character_assets", "character_images_review", "preparing_scene_assets",
    "scene_images_review", "visuals_ready",
]


def setup_page(title: str, icon: str = "🎬") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown("""<style>
    :root {
        --studio-black: #111111;
        --studio-text: #171717;
        --studio-muted: #5f5f5f;
        --studio-background: #f7f7f7;
        --studio-surface: #ffffff;
        --studio-surface-muted: #ececec;
        --studio-border: #cfcfcf;
        --studio-border-strong: #8a8a8a;
        --studio-red: #b10f1b;
        --studio-red-dark: #870b13;
        --studio-red-soft: #f8e8ea;
        --studio-focus: rgba(177, 15, 27, 0.22);
    }

    html, body, .stApp {
        background: var(--studio-background);
        color: var(--studio-text);
    }

    .stApp { color-scheme: light; }

    .stApp p, .stApp li, .stApp label, .stApp small,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--studio-text);
    }

    .stApp [data-testid="stCaptionContainer"] {
        color: var(--studio-muted) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--studio-black) !important;
        letter-spacing: -0.015em;
    }

    h1 {
        border-bottom: 3px solid var(--studio-red);
        padding-bottom: 0.35rem;
    }

    a { color: var(--studio-red-dark) !important; }
    a:hover { color: var(--studio-red) !important; }
    hr { border-color: var(--studio-border) !important; }

    [data-testid="stHeader"] {
        background: rgba(247, 247, 247, 0.94);
        border-bottom: 1px solid var(--studio-border);
    }

    /* The black project rail is the strongest brand surface. */
    [data-testid="stSidebar"] {
        background: var(--studio-black);
        border-right: 4px solid var(--studio-red);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #f5f5f5 !important;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 0.45rem;
        color: #f5f5f5 !important;
    }

    [data-testid="stSidebarNav"] a:hover { background: #292929; }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: var(--studio-red);
    }

    /* Cards, bordered containers and review surfaces. */
    [data-testid="stMetric"] {
        background: var(--studio-surface);
        border: 1px solid var(--studio-border);
        border-top: 3px solid var(--studio-red);
        padding: 12px;
        border-radius: 10px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--studio-surface);
        border-color: var(--studio-border) !important;
        border-radius: 0.65rem;
    }

    [data-testid="stForm"] {
        background: var(--studio-surface);
        border-color: var(--studio-border) !important;
    }

    [data-testid="stExpander"] {
        background: var(--studio-surface);
        border-color: var(--studio-border) !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--studio-red-dark) !important;
    }

    /* Inputs stay high-contrast in editable, disabled and focused states. */
    input, textarea, [data-baseweb="select"] > div,
    [data-baseweb="base-input"] {
        background: var(--studio-surface) !important;
        color: var(--studio-text) !important;
        -webkit-text-fill-color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }

    input:focus, textarea:focus,
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="base-input"]:focus-within {
        border-color: var(--studio-red) !important;
        box-shadow: 0 0 0 0.2rem var(--studio-focus) !important;
    }

    input:disabled, textarea:disabled,
    [data-baseweb="select"] [aria-disabled="true"] {
        background: var(--studio-surface-muted) !important;
        color: #484848 !important;
        -webkit-text-fill-color: #484848 !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background: var(--studio-surface) !important;
        color: var(--studio-text) !important;
    }

    [role="option"] { color: var(--studio-text) !important; }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: var(--studio-red-soft) !important;
        color: var(--studio-black) !important;
    }

    /* Navigation and disclosure controls. */
    [data-baseweb="tab-list"] button,
    [data-testid="stExpander"] summary,
    [data-testid="stFileUploader"] {
        color: var(--studio-text) !important;
    }

    [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: var(--studio-red-dark) !important;
        font-weight: 700;
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--studio-red) !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--studio-surface);
        border-color: var(--studio-border-strong) !important;
    }

    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        background: var(--studio-surface);
        border: 1px solid var(--studio-border);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    /* Red is reserved for action and emphasis; black anchors secondary actions. */
    [data-testid="stBaseButton-secondary"] {
        background: var(--studio-surface);
        border-color: var(--studio-black);
        color: var(--studio-black);
    }

    [data-testid="stBaseButton-secondary"]:hover {
        background: var(--studio-red-soft);
        border-color: var(--studio-red);
        color: var(--studio-red-dark);
    }

    [data-testid="stBaseButton-secondary"] p { color: inherit !important; }

    [data-testid="stBaseButton-primary"] {
        background: var(--studio-red);
        border-color: var(--studio-red);
        color: #ffffff;
        font-weight: 650;
    }

    [data-testid="stBaseButton-primary"]:hover {
        background: var(--studio-red-dark);
        border-color: var(--studio-red-dark);
    }

    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Keep the Create Project form action legible against its red background. */
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFormSubmitButton"] button p,
    [data-testid="stFormSubmitButton"] button span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    button:focus-visible, a:focus-visible {
        outline: 3px solid var(--studio-focus) !important;
        outline-offset: 2px;
    }

    button:disabled,
    [data-testid="stBaseButton-primary"]:disabled,
    [data-testid="stBaseButton-secondary"]:disabled {
        background: #dedede !important;
        border-color: #bdbdbd !important;
        color: #606060 !important;
        opacity: 1 !important;
    }

    button:disabled p { color: #606060 !important; }

    /* Alerts keep semantic meaning while matching the neutral/red system. */
    [data-testid="stAlert"] {
        background: var(--studio-surface) !important;
        border: 1px solid var(--studio-border);
        border-left: 4px solid var(--studio-red);
        color: var(--studio-text) !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div { color: var(--studio-text); }

    code, pre {
        background: #ededed !important;
        color: var(--studio-black) !important;
        border-color: var(--studio-border) !important;
    }

    ::selection { background: var(--studio-red); color: #ffffff; }

    .stage {
        padding: 7px 11px;
        border: 1px solid #d93a45;
        border-radius: 999px;
        background: var(--studio-red);
        color: #ffffff;
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stage * { color: #ffffff !important; }
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
    st.sidebar.caption(f"{project['culture_display_name']} · {project['locality']} · {project['setting']}")
    return project


def run_action(label: str, action, *, success: str = "Saved", key: str | None = None, disabled: bool = False):
    if st.button(label, key=key, disabled=disabled, type="primary"):
        try:
            result = action()
            st.success(success)
            st.rerun()
        except APIError as error:
            st.error(str(error))
