"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DocumentOut = {
  id: number;
  filename: string;
  chunk_count: number;
};

type SourceOut = {
  document: string;
  chunk_index: number;
  excerpt: string;
};

type AskResponse = {
  answer: string;
  sources: SourceOut[];
  used_llm: boolean;
};

export default function Home() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (res.ok) setDocuments(await res.json());
    } catch {
      // Backend not reachable yet; the user will see this via the ask/upload errors.
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
      setFile(null);
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setAsking(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4 }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Question failed");
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Question failed");
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="container">
      <h1>RAG Document Assistant</h1>
      <p className="subtitle">
        Upload a document, then ask questions grounded in its content.
      </p>

      <div className="card">
        <h2>1. Upload a document (.pdf or .txt)</h2>
        <div className="row">
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>

        {documents.length > 0 && (
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.id}>
                {doc.filename} <span className="badge">{doc.chunk_count} chunks</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <h2>2. Ask a question</h2>
        <div className="row">
          <input
            type="text"
            placeholder="What does this document say about...?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          />
          <button onClick={handleAsk} disabled={!question.trim() || asking}>
            {asking ? "Thinking..." : "Ask"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {result && (
          <>
            <p className="answer">
              {result.answer}
              <span className="badge">{result.used_llm ? "LLM answer" : "extractive"}</span>
            </p>
            {result.sources.map((s, i) => (
              <div className="source" key={i}>
                <strong>
                  {s.document} - chunk {s.chunk_index}
                </strong>
                <div>{s.excerpt}</div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
