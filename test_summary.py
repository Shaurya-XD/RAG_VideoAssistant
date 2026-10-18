from pathlib import Path

from core.summarise import summarise_transcript


def main():
    # Read the transcript saved by our earlier transcription test.
    transcript = Path("transcript.txt").read_text(encoding="utf-8")

    print("Generating summary...")

    # Omit language="English" to keep the transcript's language.
    summary = summarise_transcript(transcript, language="English")

    # Save the summary separately from the original transcript.
    Path("summary.txt").write_text(summary, encoding="utf-8")

    print("\n--- Summary ---\n")
    print(summary)
    print("\nSaved to summary.txt")


if __name__ == "__main__":
    main()