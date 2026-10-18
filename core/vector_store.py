import os
import weakref
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import validate_transcript


load_dotenv()

CHROMA_DIR = "chroma_db"


def _delete_collection(client, collection_name):
    """Release only this workspace's collection when its store is collected."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Cleanup must not interrupt a reset or application shutdown.
        pass


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


def build_vector_store(transcript: str, persist_directory: str | None = None) -> Chroma:
    """Create an isolated video index; keep it temporary unless saving explicitly."""
    validate_transcript(transcript)

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

    # Every build gets a unique name, including builds from different users.
    collection_name = f"video_{uuid4().hex}"
    settings = Settings(anonymized_telemetry=False)
    client = (
        chromadb.PersistentClient(path=persist_directory, settings=settings)
        if persist_directory
        else chromadb.EphemeralClient(settings=settings)
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
    )

    try:
        # Small batches avoid a single oversized embedding request.
        for start in range(0, len(docs), 32):
            vector_store.add_documents(docs[start:start + 32])
    except Exception:
        _delete_collection(client, collection_name)
        raise

    if not persist_directory:
        # The retriever keeps this store alive while its session chain is in use.
        # Resetting/discarding the workspace lets Python clean up its collection.
        weakref.finalize(vector_store, _delete_collection, client, collection_name)

    print(f"Stored {len(docs)} chunks.")
    return vector_store


def load_vector_store(collection_name: str, persist_directory: str = CHROMA_DIR) -> Chroma:
    """Open an explicitly saved collection, never a shared 'current video'."""

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
        client_settings=Settings(anonymized_telemetry=False),
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
