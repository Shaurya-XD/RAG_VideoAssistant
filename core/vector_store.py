import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "video_transcript"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL")

    if not api_key or not model_name:
        raise ValueError(
            "Set GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL in .env."
        )

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )


def build_vector_store(transcript: str) -> Chroma:

    if not transcript.strip():
        raise ValueError("The transcript is empty.")

    print("Building vector store...")

    # Split text into overlapping chunks measured in characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_text(transcript)

    # Store each chunk with its position in the transcript.
    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # Clear the previous video's collection before adding the new one.
    existing_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    existing_store.delete_collection()

    # Generate embeddings and save the new index locally.
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Stored {len(docs)} chunks.")
    return vector_store


def load_vector_store() -> Chroma:
    """Open the current video's saved index."""

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

    if not vector_store.get(limit=1)["ids"]:
        raise ValueError("No transcript has been indexed yet.")

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """Retrieve the k most similar chunks for a question."""

    if k <= 0:
        raise ValueError("k must be positive.")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )