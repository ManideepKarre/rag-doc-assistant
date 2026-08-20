"""
Core RAG (Retrieval-Augmented Generation) utilities: text chunking and prompt building.

These are kept as pure functions (no DB/network calls) so they are easy to unit test
in isolation from the FastAPI app, the database, and the embedding model.
"""
from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    A simple sliding-window splitter. It tries to break on whitespace near the
    boundary so words aren't cut in half, but falls back to a hard cut if no
    whitespace is found nearby.

    Args:
        text: The full document text.
        chunk_size: Target maximum size (in characters) of each chunk.
        overlap: Number of characters of overlap between consecutive chunks,
            which helps preserve context that would otherwise be split across
            a chunk boundary.

    Returns:
        A list of text chunks. Empty/whitespace-only input returns [].
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Try to break on the last whitespace within the window so we
            # don't split a word in half.
            window = text[start:end]
            last_space = window.rfind(" ")
            if last_space != -1 and last_space > chunk_size * 0.5:
                end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Move the window forward, backing up by `overlap` characters.
        start = max(end - overlap, start + 1)

    return chunks


def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Build a grounded RAG prompt from a question and retrieved context chunks."""
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        "You are a helpful assistant answering questions using ONLY the context "
        "below. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources like [Source 1] where relevant):"
    )
