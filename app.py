"""Streamlit interface for the YouTube Video Assistant.

Run from the project folder with: streamlit run app.py
"""

import os
import re
import hmac
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
    initial_sidebar_state="expanded",
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
    /* Sidebar: product identity, workspace, and the settings that matter. */
    [data-testid="stSidebar"] {
        background: #11141c; border-right: 1px solid #252a36;
        min-width: 245px !important; max-width: 245px !important;
    }
    [data-testid="stSidebarHeader"] { height: 2.6rem; min-height: 2.6rem; padding: .25rem .7rem; }
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"] span { color: #9ba6bd !important; }
    [data-testid="stSidebarUserContent"] { padding: 1.1rem 1.15rem; }
    .brand { display: flex; align-items: center; gap: .7rem; margin: .3rem 0 2rem; }
    .brand-mark { width: 35px; height: 35px; border-radius: 10px; display: grid;
        place-items: center; background: #a99aff; color: #151224; font-size: 19px; }
    .brand-name { color: #f2f3fa; font-size: 1.05rem; font-weight: 650; letter-spacing: -.025em; }
    .brand-subtitle { color: #7f8ba4; font-size: .68rem; margin-top: 1px; }
    .side-label { color: #77839b; font-size: .65rem; font-weight: 650;
        letter-spacing: .12em; margin: 1.4rem 0 .6rem; }
    .side-active { background: #222137; border: 1px solid #35304d; border-radius: 8px;
        padding: .65rem .8rem; color: #d1c7ff; font-size: .85rem; }
    .side-active span { margin-right: .65rem; color: #b8aaff; }
    .side-guide { border-left: 1px solid #303648; margin: .9rem 0 1.5rem .4rem; padding-left: .9rem; }
    .side-guide div { color: #929eb6; font-size: .79rem; padding: .35rem 0; }
    .session-note { color: #7e8aa2; font-size: .73rem; line-height: 1.7;
        border-top: 1px solid #262c39; padding-top: 1rem; margin-top: 1.2rem; }
    .session-dot { color: #89ceb3; margin-right: .4rem; }
    /* Page hierarchy. */
    .workspace-bar { display: flex; justify-content: space-between; align-items: center;
        color: #808ca5; font-size: .7rem; letter-spacing: .04em; margin: .1rem 0 1.6rem; }
    .workspace-bar b { color: #b5bfd2; font-weight: 500; }
    .status-badge { display: inline-flex; align-items: center; gap: .4rem;
        color: #aeb9ce; background: #171d26; padding: .3rem .7rem;
        border: 1px solid #2a3441; border-radius: 20px; font-size: .67rem; }
    .status-badge i { width: 5px; height: 5px; border-radius: 50%; background: #8bd4b3; }
    .hero { padding: .25rem 0 1.1rem; }
    .hero .eyebrow { color: #b0a1ff; font-size: .67rem; font-weight: 650;
        letter-spacing: .14em; margin-bottom: 1rem; }
    .hero h1 { margin: 0; padding: 0; line-height: 1.18; font-size: clamp(32px, 3.8vw, 48px) !important;
        max-width: 760px; font-weight: 650 !important; }
    .hero h1 [data-heading-text] { color: #edf0f8; }
    .hero h1 [data-heading-text] span { color: #b5a6ff; }
    .hero p { color: #9ca8bf; font-size: .95rem; line-height: 1.75;
        max-width: 640px; margin-top: 1rem; }
    .panel-heading { color: #cbd3e3; font-size: .7rem; font-weight: 600;
        letter-spacing: .1em; margin-bottom: .8rem; }
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
    /* Empty state cards explain capabilities without adding more controls. */
    .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
    .feature { background: #141822; border: 1px solid #272f40; border-radius: 12px;
        padding: 1.35rem; min-height: 190px; }
    .feature-icon { display: grid; place-items: center; width: 36px; height: 36px;
        border-radius: 9px; margin-bottom: 1.1rem; }
    .feature-icon svg { width: 19px; height: 19px; }
    .violet { color: #b8a8ff; background: #2a2341; }
    .teal { color: #86d8c7; background: #19312f; }
    .amber { color: #e7c18a; background: #332b21; }
    .feature h3 { font-size: .94rem !important; font-weight: 600; padding: 0; margin-bottom: .5rem; }
    .feature p { color: #929eb6; font-size: .81rem; line-height: 1.7; margin: 0; }
    .section-kicker { font-size: .67rem; color: #7e8ba5; font-weight: 600;
        letter-spacing: .12em; margin-top: 1.2rem; }
    .workspace-hint { border: 1px dashed #333b4e; border-radius: 12px; padding: 1.05rem 1.25rem;
        display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
    .workspace-hint strong { color: #b9c3d7; font-size: .81rem; font-weight: 500; }
    .workspace-hint span { color: #7c89a2; font-size: .74rem; }
    /* Result workspace and chat. */
    [data-testid="stMetric"] { background: #151923; border: 1px solid #2a3141;
        padding: 1rem 1.15rem; border-radius: 11px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #98a5bf !important; font-size: .75rem; }
    [data-testid="stMetricValue"] { color: #e7ecf9 !important; font-size: 1.45rem; }
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
    .footer-note { color: #78859e; font-size: .72rem; padding: 1.3rem 0 .2rem;
        margin-top: 1rem; border-top: 1px solid #252d3d; }
    @media (max-width: 900px) {
        .block-container { padding: 1.5rem 1.4rem; }
        .features { gap: .65rem; }
        .feature { padding: 1rem; }
    }
    @media (max-width: 640px) {
        .features { grid-template-columns: 1fr; }
        .feature { min-height: 0; }
        .feature-icon { float: left; margin: 0 .9rem 0 0; }
        .hero h1 { font-size: 34px !important; }
        .workspace-hint { align-items: flex-start; flex-direction: column; gap: .35rem; }
        .workspace-bar { margin-bottom: 1rem; }
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








# Optional password gate for a shared demo using the owner's API quota.
app_password = os.getenv("APP_PASSWORD")
if app_password and not st.session_state.get("authenticated", False):
    st.title("Video Notes")
    with st.form("sign_in"):
        password = st.text_input("App password", type="password")
        sign_in = st.form_submit_button("Open workspace", type="primary")
    if sign_in:
        if hmac.compare_digest(password.encode(), app_password.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


if "workspace" not in st.session_state:
    st.session_state.workspace = empty_workspace()
workspace = st.session_state.workspace

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">▷</div>'
        '<div><div class="brand-name">Video Notes</div>'
        '<div class="brand-subtitle">A clearer way to watch.</div></div></div>'
        '<div class="side-label">WORKSPACE</div>'
        '<div class="side-active" aria-current="page"><span>◈</span> Video workspace</div>',
        unsafe_allow_html=True,
    )
    if st.button("＋ New video", use_container_width=True):
        st.session_state.workspace = empty_workspace()
        st.rerun()
    st.markdown(
        '<div class="side-label">HOW IT WORKS</div>'
        '<div class="side-guide"><div>01 &nbsp; Add a YouTube link</div>'
        '<div>02 &nbsp; Read the essential ideas</div>'
        '<div>03 &nbsp; Ask your own questions</div></div>'
        '<div class="side-label">PREFERENCES</div>',
        unsafe_allow_html=True,
    )
    selected_language = st.selectbox("Summary language", ["Original language", "English", "Hindi"])
    language = "the same language as the source" if selected_language == "Original language" else selected_language
    st.caption("Answers follow the language of your question.")
    st.markdown(
        '<div class="session-note"><span class="session-dot">●</span> Your session workspace'
        '<br>Download your notes to keep them after this session.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="workspace-bar"><span>WORKSPACE &nbsp; / &nbsp; <b>Video Notes</b></span>'
    '<span class="status-badge"><i></i> A space for your ideas</span></div>',
    unsafe_allow_html=True,
)
if workspace["transcript"]:
    st.markdown(
        '<div class="hero"><div class="eyebrow">YOUR VIDEO, UNDERSTOOD</div>'
        '<h1>One video. <span>Every insight.</span></h1>'
        '<p>Your transcript, essential ideas, and a place to ask better questions.</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="hero"><div class="eyebrow">LESS REPLAYING. MORE UNDERSTANDING.</div>'
        '<h1>A clearer view of<br><span>every video.</span></h1>'
        '<p>Bring a YouTube link. Leave with the transcript, the important ideas,'
        ' and answers to the questions that matter to you.</p></div>',
        unsafe_allow_html=True,
    )

with st.form("video_form"):
    st.markdown('<div class="panel-heading">START WITH A YOUTUBE VIDEO</div>', unsafe_allow_html=True)
    link_column, action_column = st.columns([4.6, 1.4], gap="small", vertical_alignment="center")
    with link_column:
        url = st.text_input(
            "YouTube video URL", placeholder="Paste your YouTube link here…", label_visibility="collapsed"
        )
    with action_column:
        process_video = st.form_submit_button("Analyze video →", type="primary", use_container_width=True)
    st.caption("Public videos · Hindi & English · Start with a short video")

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
                st.success("Your transcript is ready. Generate a summary or start exploring below.")
            except Exception as error:
                show_service_error(error, "Transcription")

if not workspace["transcript"]:
    # Static illustrations are inline SVG; no external assets or fonts are loaded.
    st.markdown(
        '<div class="section-kicker">ONE LINK. THREE WAYS TO UNDERSTAND.</div>'
        '<div class="features">'
        '<div class="feature"><div class="feature-icon violet">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">'
        '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/></svg></div>'
        '<h3>The full story</h3><p>A readable transcript in the original language. '
        'Find the details you want to revisit.</p></div>'
        '<div class="feature"><div class="feature-icon teal">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">'
        '<path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4Z"/></svg></div>'
        '<h3>Only the essentials</h3><p>A concise summary and clear key points. '
        'Spend your time on the ideas, not the replay.</p></div>'
        '<div class="feature"><div class="feature-icon amber">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">'
        '<path d="M20 11.5a8 8 0 0 1-8 8H4l1.5-4A8 8 0 1 1 20 11.5Z"/>'
        '<path d="M8 10h8M8 13h5"/></svg></div>'
        '<h3>A conversation with context</h3><p>Ask focused questions and get answers '
        'grounded in what the video actually says.</p></div></div>'
        '<div class="workspace-hint"><strong>Your next useful idea is one link away.</strong>'
        '<span>No video downloads. Just paste, read, and explore.</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.caption(f"Current video: {workspace['url']}")
    words, summary_status, question_status = st.columns(3)
    words.metric("Transcript words", f"{len(workspace['transcript'].split()):,}")
    summary_status.metric("Summary", "Ready" if language in workspace["summaries"] else "Not generated")
    question_status.metric("Video questions", "Ready" if workspace["rag_chain"] else "Not prepared")

    overview_tab, transcript_tab, questions_tab = st.tabs(["Summary & key points", "Transcript", "Ask questions"])

    with overview_tab:
        st.subheader("The essential ideas")
        st.caption(f"Summary in {selected_language.lower()}. Generate it once and revisit it anytime in this session.")
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
            st.caption("Extracted from the summary.")
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
            with st.container(border=True):
                st.markdown("**A shorter path to the main ideas.**")
                st.caption("Generate a summary to bring the video's explanations and conclusions into focus.")

    with transcript_tab:
        st.subheader("The full transcript")
        st.caption("Transcribed in the video's original language. Transcription can contain errors.")
        st.text_area("Transcript", workspace["transcript"], height=460, disabled=True, label_visibility="collapsed")
        st.download_button("Download transcript", workspace["transcript"], "transcript.txt", "text/plain", on_click="ignore")
        with st.expander("Watch the source video"):
            st.video(workspace["url"])

    with questions_tab:
        st.subheader("What would you like to understand?")
        st.caption("Ask a complete question each time. Earlier messages stay visible, but aren't used as conversation memory.")
        if workspace["rag_chain"] is None:
            st.write("Prepare this video's transcript for questions once, then reuse it throughout your session.")
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

st.markdown(
    '<div class="footer-note">Video Notes &nbsp; · &nbsp; Built around your curiosity.'
    ' &nbsp; Answers reflect the transcript; verify important details with the source.</div>',
    unsafe_allow_html=True,
)
