"""Streamlit interface for the YouTube Video Assistant.

Run from the project folder with: streamlit run app.py
"""

import os
import re
import logging
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv

from core.rag_engine import ask_question, build_rag_chain, generate_key_points
from core.settings import MAX_CHAT_MESSAGES, MAX_QUESTION_CHARS
from core.transcriber import transcribe_video


load_dotenv()
st.set_page_config(
    page_title="Video Notes — Your video, understood",
    page_icon="◈",
    layout="wide",
)

# Styling stays in this UI file. User input is never inserted into HTML.
st.markdown(
    """
    <style>
    /* Explicit foregrounds and surfaces prevent clashes with either host theme. */
    .stApp {
        --vn-bg: #0d1017; --vn-panel: #151923; --vn-border: #292f3e;
        --vn-text: #edf0f8; --vn-muted: #9ba6bd; --vn-accent: #a99aff;
        background: var(--vn-bg); color: var(--vn-text);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #0d1017; }
    [data-testid="stHeader"] { background: #0d1017; }
    .block-container { max-width: 1140px; padding: 2.4rem 3rem 2rem; }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 1.15rem; }
    h1, h2, h3, h4 { color: #edf0f8 !important; letter-spacing: -.035em; }
    h1 { font-size: 2.65rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.25rem !important; }
    [data-testid="stMarkdownContainer"] { color: #d9dfec; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.7; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #9ba6bd !important; font-size: .8rem;
    }
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: #cad2e4 !important;
    }
    a { color: #baadff; }
    hr { border-color: #262c39 !important; }
    [data-testid="stForm"] { background: #151923; border: 1px solid #30364a !important;
        border-radius: 14px !important; padding: 1.2rem 1.3rem !important;
        box-shadow: 0 12px 40px #00000014; }
    [data-testid="stTextInputRootElement"] { background: #0f131c !important;
        border: 1px solid #343b4f !important; border-radius: 9px !important; }
    input, textarea { color: #e9edf8 !important; -webkit-text-fill-color: #e9edf8 !important;
        background: #0f131c !important; caret-color: #b9acff; }
    input::placeholder, textarea::placeholder {
        color: #77849e !important; -webkit-text-fill-color: #77849e !important; opacity: 1;
    }
    [data-testid="stTextInput"] input { min-height: 46px; font-size: .86rem; }
    [data-baseweb="select"] > div { background: #191e2a !important;
        color: #dfe5f3 !important; border-color: #32394b !important; border-radius: 8px; }
    [data-baseweb="select"] span { color: #dfe5f3 !important; }
    [data-testid="stSelectbox"] [role="group"] { background: #191e2a !important;
        border: 1px solid #32394b !important; border-radius: 8px !important; }
    [data-testid="stSelectbox"] [role="group"]:focus-within { border-color: #a99aff !important; }
    [data-testid="stSelectbox"] button { background: #191e2a !important;
        color: #cbd3e5 !important; border: none !important; }
    [data-testid="stSelectbox"] input { background: #191e2a !important; }
    [data-baseweb="popover"] [role="listbox"] { background: #191e2a !important; }
    [role="listbox"] { background: #191e2a !important; color: #dfe5f3 !important; }
    [role="option"] { background: #191e2a !important; color: #dfe5f3 !important; }
    [role="option"]:hover { background: #2b2943 !important; }
    /* Native controls stay keyboard-accessible; only their appearance changes. */
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: #1a2030 !important; border: 1px solid #343c50 !important;
        border-radius: 8px !important; color: #dce3f4 !important;
        font-size: .82rem; font-weight: 500; min-height: 42px; transition: border-color .15s;
    }
    .stButton button:hover, .stDownloadButton button:hover { border-color: #8f80cf !important; }
    button[kind="primary"], button[kind="primaryFormSubmit"],
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
        background: #b4a4ff !important; border-color: #b4a4ff !important;
        color: #171226 !important; font-weight: 650 !important;
    }
    button[kind="primary"] p, button[kind="primaryFormSubmit"] p,
    [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {
        color: #171226 !important;
    }
    button:disabled { opacity: .45 !important; }
    button:focus-visible, input:focus-visible { outline: 2px solid #b7a8ff !important; outline-offset: 3px; }
    /* Result tabs and chat. */
    [data-baseweb="tab-list"] { background: #141923; padding: 5px; border: 1px solid #293143;
        border-radius: 10px; gap: .3rem; margin: .35rem 0 1rem; }
    [data-baseweb="tab"] { background: transparent !important; color: #99a6bf !important;
        padding: .5rem 1rem !important; border-radius: 7px; height: auto; }
    [data-baseweb="tab"] p { color: inherit !important; font-size: .83rem; }
    [data-baseweb="tab"][aria-selected="true"] { background: #302b48 !important; color: #d6ccff !important; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { background: transparent !important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #2a3141 !important; border-radius: 12px; }
    [data-testid="stTextArea"] textarea { border: 1px solid #30384b; border-radius: 10px;
        font-size: .87rem; line-height: 1.8; padding: 1rem; opacity: 1; }
    [data-testid="stChatMessage"] { background: #171c28; border: 1px solid #2a3244; border-radius: 12px; }
    [data-testid="stChatInput"] { background: #191e2a !important; border: 1px solid #39415a !important; border-radius: 12px; }
    [data-testid="stChatInput"] textarea { background: transparent !important; }
    [data-testid="stExpander"] details { border-color: #30384b !important; background: #151923; }
    [data-testid="stAlert"] { border-radius: 10px; }
    @media (max-width: 640px) {
        .block-container { padding: 1.5rem 1rem; }
        h1 { font-size: 2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def empty_workspace():
    """Keep generated content and resources separate for each browser session."""
    return {
        "url": "",
        "transcript": "",
        "summaries": {},
        "key_points": {},
        "messages": [],
        "rag_chain": None,
    }


def canonical_youtube_url(value: str) -> str:
    """Validate common public-video URL formats and return one canonical URL."""
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    parts = parsed.path.strip("/").split("/")
    video_id = ""

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Paste a complete YouTube URL beginning with https://.")
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parts[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            video_id = parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a YouTube video link, rather than a channel or playlist link.")
    return f"https://www.youtube.com/watch?v={video_id}"


def settings_available(names):
    """Explain missing configuration without displaying secret values."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        st.error("Missing configuration: " + ", ".join(missing))
        st.caption("Add these settings to your local .env or Streamlit app secrets.")
    return not missing


def show_service_error(error, action):
    """Show useful errors without exposing provider responses or API keys."""
    # Log the exception type without exposing provider responses or keys.
    logging.getLogger(__name__).error("%s failed (%s)", action, type(error).__name__)
    message = str(error).lower()
    if any(word in message for word in ("429", "rate limit", "resource_exhausted")):
        st.error(f"{action} reached the API usage limit. Wait for your quota to reset, then retry.")
    elif any(word in message for word in ("401", "403", "api key", "unauthenticated")):
        st.error(f"{action} could not authenticate. Check the configured API key and model access.")
    elif any(word in message for word in ("tokens exhausted", "output limit", "did not complete")):
        st.error(f"{action} could not produce a complete result. Try a shorter video or retry.")
    else:
        st.error(f"{action} failed. Check model settings and your API quota, then retry.")


if "workspace" not in st.session_state:
    st.session_state.workspace = empty_workspace()
workspace = st.session_state.workspace

# Keep the workspace controls on the page; there is no sidebar.
heading, reset = st.columns([5, 1], vertical_alignment="center")
with heading:
    st.title("Video Notes")
with reset:
    if st.button("＋ New video", use_container_width=True):
        st.session_state.workspace = empty_workspace()
        st.rerun()
st.caption("Transcribe, summarise, and ask questions about a YouTube video.")

with st.form("video_form"):
    link_column, action_column = st.columns([4.6, 1.4], gap="small", vertical_alignment="center")
    with link_column:
        url = st.text_input(
            "YouTube video URL", placeholder="Paste your YouTube link here…", label_visibility="collapsed"
        )
    with action_column:
        process_video = st.form_submit_button("Analyze video →", type="primary", use_container_width=True)
    st.caption("Public YouTube videos only.")

if not workspace["transcript"]:
    # Brief feature descriptions keep the empty workspace informative.
    transcript_info, summary_info, question_info = st.columns(3)
    with transcript_info:
        st.markdown("**Transcript**")
        st.caption("Read and download the video's spoken content.")
    with summary_info:
        st.markdown("**Summary & key points**")
        st.caption("Get the main ideas in English, Hindi, or the original language.")
    with question_info:
        st.markdown("**Ask questions**")
        st.caption("Find answers grounded in the video.")

if process_video:
    try:
        validated_url = canonical_youtube_url(url)
    except ValueError as error:
        st.error(str(error))
    else:
        if validated_url == workspace["url"] and workspace["transcript"]:
            st.info("This video is already in your workspace.")
        elif settings_available(["GEMINI_API_KEY", "GEMINI_MODEL"]):
            try:
                with st.spinner("Reading the video and transcribing its speech…"):
                    transcript = transcribe_video(validated_url)
                # Keep the old workspace if the new transcription fails.
                workspace = empty_workspace()
                workspace.update(url=validated_url, transcript=transcript)
                st.session_state.workspace = workspace
                st.success("Transcript ready.")
            except Exception as error:
                show_service_error(error, "Transcription")

if workspace["transcript"]:
    overview_tab, transcript_tab, questions_tab = st.tabs(["Summary & key points", "Transcript", "Ask questions"])

    with overview_tab:
        selected_language = st.selectbox("Summary language", ["Original language", "English", "Hindi"])
        language = "the same language as the source" if selected_language == "Original language" else selected_language
        if st.button("Generate summary", type="primary", disabled=language in workspace["summaries"]):
            if settings_available(["GROQ_API_KEY", "GROQ_MODEL"]):
                try:
                    with st.spinner("Summarising the transcript…"):
                        # Load the tokenizer only when summarisation is requested.
                        from core.summarise import summarise_transcript

                        workspace["summaries"][language] = summarise_transcript(
                            workspace["transcript"], language=language
                        )
                    st.rerun()
                except Exception as error:
                    show_service_error(error, "Summarisation")

        summary = workspace["summaries"].get(language)
        if summary:
            with st.container(border=True):
                st.markdown(summary)
            st.download_button("Download summary", summary, "summary.txt", "text/plain", on_click="ignore")
            st.divider()
            st.subheader("Key points")
            if st.button("Extract key points", disabled=language in workspace["key_points"]):
                try:
                    with st.spinner("Pulling out the key points…"):
                        workspace["key_points"][language] = generate_key_points(summary, language)
                    st.rerun()
                except Exception as error:
                    show_service_error(error, "Key point extraction")
            if language in workspace["key_points"]:
                points = workspace["key_points"][language]
                with st.container(border=True):
                    st.markdown(points)
                st.download_button("Download key points", points, "key_points.txt", "text/plain", on_click="ignore")
        else:
            st.caption("Generate a summary to see the key ideas.")

    with transcript_tab:
        st.text_area("Transcript", workspace["transcript"], height=460, disabled=True, label_visibility="collapsed")
        st.download_button("Download transcript", workspace["transcript"], "transcript.txt", "text/plain", on_click="ignore")
        with st.expander("Watch the source video"):
            st.video(workspace["url"])

    with questions_tab:
        st.caption("Answers use video excerpts. Ask each question in full.")
        if workspace["rag_chain"] is None:
            if st.button("Prepare video questions", type="primary"):
                if settings_available(["GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL", "GROQ_API_KEY", "GROQ_MODEL"]):
                    try:
                        with st.spinner("Preparing searchable transcript passages…"):
                            workspace["rag_chain"] = build_rag_chain(workspace["transcript"])
                        st.rerun()
                    except Exception as error:
                        show_service_error(error, "Question preparation")

        for message in workspace["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # A chat input inside a tab remains inline within that tab.
        question = st.chat_input(
            "Ask something about this video…",
            disabled=workspace["rag_chain"] is None,
            max_chars=MAX_QUESTION_CHARS,
        )
        if question and question.strip():
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Finding relevant passages…"):
                        answer = ask_question(workspace["rag_chain"], question)
                    st.markdown(answer)
                workspace["messages"].extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
                workspace["messages"] = workspace["messages"][-MAX_CHAT_MESSAGES:]
            except Exception as error:
                show_service_error(error, "Answer generation")
