"""
RAG Document Assistant - FastAPI backend.

Endpoints:
  POST /documents/upload  - upload a .pdf or .txt file; it is chunked, embedded,
                             and stored in Postgres (with pgvector) for retrieval.
  GET  /documents         - list uploaded documents.
  POST /ask                - ask a question; the top-matching chunks are retrieved
                             via cosine similarity and used to ground an answer.
  GET  /health             - simple liveness check.

Embeddings are generated locally with sentence-transformers (no API key needed).
Answer generation uses the OpenAI API if OPENAI_API_KEY is set; otherwise it falls
back to an "extractive" answer built directly from the retrieved chunks, so the
app is fully runnable out of the box with zero paid API keys.
"""
import io
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Chunk, Document, get_db, init_db
from rag_core import build_prompt, chunk_text

load_dotenv()

app = FastAPI(title="RAG Document Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load the embedding model (avoids the ~80MB download at import time,
    e.g. during unit tests that don't need it)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DocumentOut(BaseModel):
    id: int
    filename: str
    chunk_count: int

    class Config:
        from_attributes = True


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class SourceOut(BaseModel):
    document: str
    chunk_index: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
    used_llm: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    raw = await file.read()
    filename = file.filename or "untitled"

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    model = get_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)

    document = Document(filename=filename)
    db.add(document)
    db.flush()  # get document.id before inserting chunks

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=idx,
                content=chunk,
                embedding=embedding.tolist(),
            )
        )
    db.commit()

    return DocumentOut(id=document.id, filename=document.filename, chunk_count=len(chunks))


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    documents = db.execute(select(Document)).scalars().all()
    return [
        DocumentOut(id=d.id, filename=d.filename, chunk_count=len(d.chunks)) for d in documents
    ]


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, db: Session = Depends(get_db)):
    model = get_embedding_model()
    question_embedding = model.encode(payload.question, normalize_embeddings=True).tolist()

    results = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(Chunk.embedding.cosine_distance(question_embedding))
        .limit(payload.top_k)
    ).all()

    if not results:
        raise HTTPException(status_code=404, detail="No documents have been uploaded yet")

    sources = [
        SourceOut(document=doc.filename, chunk_index=chunk.chunk_index, excerpt=chunk.content)
        for chunk, doc in results
    ]

    answer, used_llm = _generate_answer(payload.question, [s.excerpt for s in sources])
    return AskResponse(answer=answer, sources=sources, used_llm=used_llm)


def _generate_answer(question: str, context_chunks: List[str]) -> tuple[str, bool]:
    """Generate an answer via the OpenAI API if configured, else fall back to a
    simple extractive summary built from the retrieved chunks."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = build_prompt(question, context_chunks)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip(), True
        except Exception as exc:  # pragma: no cover - network/credential errors
            fallback = "\n\n".join(context_chunks[:2])
            return (
                f"(LLM call failed: {exc}. Showing the most relevant excerpts instead.)\n\n"
                f"{fallback}",
                False,
            )

    # No API key configured: return the most relevant excerpts directly.
    fallback = "\n\n".join(context_chunks[:2])
    return (
        "No OPENAI_API_KEY is configured, so here are the most relevant excerpts "
        f"found for your question:\n\n{fallback}",
        False,
    )
