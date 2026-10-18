"""Small limits for a shared demo deployment."""

# Character limits bound per-session storage and API request sizes.
# They are not video-duration limits or exact provider token counts.
MAX_TRANSCRIPT_CHARS = 100_000
MAX_QUESTION_CHARS = 2_000
MAX_CHAT_MESSAGES = 40


def validate_transcript(transcript: str) -> None:
    if not transcript.strip():
        raise ValueError("The transcript is empty.")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise ValueError("Transcript exceeds the app limit. Please use a shorter video.")
