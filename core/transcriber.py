import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.settings import validate_transcript


load_dotenv()


def transcribe_video(youtube_url: str) -> str:

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key or not model_name:
        raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL in .env.")

    if not youtube_url.strip():
        raise ValueError("Please provide a YouTube URL.")

    model = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_output_tokens=16384,
        timeout=180,
        max_retries=1,
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

    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason not in {"STOP", "stop"}:
        raise RuntimeError("Gemini did not complete the transcript. Try a shorter video.")

    # Extract plain text
    transcript = response.text.strip()

    validate_transcript(transcript)

    return transcript
