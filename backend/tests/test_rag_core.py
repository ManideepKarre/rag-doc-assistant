"""Unit tests for rag_core.py. Pure functions only - no DB, no ML model, no
network access required, so these run fast in any environment (incl. CI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from rag_core import build_prompt, chunk_text

def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []

def test_chunk_text_short_text_is_a_single_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == [text]

def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = "word " * 1000  # 5000 characters
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 800 + 1  # allow trailing partial word at boundary

def test_chunk_text_consecutive_chunks_overlap():
    text = "abcdefgh " * 200
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # Some suffix of chunk N should reappear as a prefix-ish region of chunk N+1
    assert any(chunks[0][-20:] in chunks[1] for _ in [0])

def test_chunk_text_invalid_arguments_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=-1)

def test_build_prompt_includes_question_and_numbered_sources():
    prompt = build_prompt("What is RAG?", ["Chunk A", "Chunk B"])
    assert "What is RAG?" in prompt
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    assert "Chunk A" in prompt
    assert "Chunk B" in prompt
