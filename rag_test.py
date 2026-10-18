from pathlib import Path

from core.rag_engine import ask_question, build_rag_chain


def main():
    transcript = Path("transcript.txt").read_text(encoding="utf-8")

    # Build once before the question loop, not for every question.
    rag_chain = build_rag_chain(transcript)

    print("\nAsk questions about the video. Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()