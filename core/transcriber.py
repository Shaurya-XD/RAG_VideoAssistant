import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def transcribe_video(youtube_url: str) -> str:
    """Generate a timestamped transcript from a public YouTube URL."""

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key or not model_name:
        raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL in .env.")

    if not youtube_url.strip():
        raise ValueError("Please provide a YouTube URL.")

    model = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
    )

    # Send the YouTube URL as media, together with our instructions.
    message = HumanMessage(
        content=[
            {
                "type": "media",
                "file_uri": youtube_url.strip(),
            },
            {
                "type": "text",
                "text": (
                    "Transcribe all speech faithfully in its original language, preserving "
                    "names, numbers, technical terms, and language switches. "
                    "Do not paraphrase or correct the speaker's factual claims. "
                    "Mark unclear speech [inaudible] instead of guessing. "
                    "Group related sentences into paragraphs separated by a blank line. "
                    "Return only the transcript, without timestamps"
                ),
            },
        ]
    )

    # Invoke sends the request and waits for response
    response = model.invoke([message])

    if response.response_metadata.get("finish_reason") == "MAX_TOKENS":
        raise RuntimeError("Tokens Exhausted")

    # Extract plain text
    transcript = response.text.strip()

    if not transcript:
        raise RuntimeError("Gemini returned an empty transcription.")

    return transcript