from pathlib import Path

from core.transcriber import transcribe_video


def main():
    # Ask for the video URL in the terminal.
    # youtube_url = input("Enter a YouTube URL: ").strip()
    youtube_url = "https://www.youtube.com/watch?v=_5p1_TNSWqQ"

    print("Generating transcription...")
    transcript = transcribe_video(youtube_url)
 
    # Save the result as UTF-8 to support different languages.
    output_path = Path("transcript.txt")
    output_path.write_text(transcript, encoding="utf-8")

    print("\n--- Transcript ---\n")
    print(transcript)
    print(f"\nSaved to: {output_path.resolve()}")


# Run this function only when test.py is executed directly.
if __name__ == "__main__":
    main()