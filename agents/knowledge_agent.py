import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
INDEX_DIR = Path(__file__).resolve().parent.parent / "logs" / "faiss_index"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".csv", ".json"}

# Singleton cache for the embedding model so it loads only once per session
_EMBEDDER = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def extract_text_from_file(file_path: Path) -> str:
    """Extracts plain text from MD, TXT, PDF, DOCX, CSV, or JSON files."""
    ext = file_path.suffix.lower()
    try:
        if ext in {".md", ".txt"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            from docx import Document
            doc = Document(str(file_path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        elif ext == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading {file_path.name}: {e}"
    return ""


def chunk_document(text: str, ext: str) -> List[str]:
    """Splits document text into logical sections or ~800-character blocks."""
    if ext == ".md" and "\n## " in text:
        return [s.strip() for s in re.split(r"(?=\n## )", text) if s.strip()]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < 800:
            current = f"{current}\n\n{p}".strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def _compute_kb_signature() -> str:
    """Computes an MD5 fingerprint of all files in knowledge_base/ to auto-detect edits or new uploads."""
    hasher = hashlib.md5()
    if not KB_DIR.exists():
        return ""
    for file_path in sorted(KB_DIR.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            stat = file_path.stat()
            hasher.update(f"{file_path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
    return hasher.hexdigest()


def build_or_load_faiss_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Builds a cosine-similarity FAISS index or loads the cached index if no files changed."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_file = INDEX_DIR / "kb.index"
    meta_file = INDEX_DIR / "kb_meta.pkl"
    sig_file = INDEX_DIR / "kb_sig.txt"

    current_sig = _compute_kb_signature()

    if index_file.exists() and meta_file.exists() and sig_file.exists():
        if sig_file.read_text(encoding="utf-8").strip() == current_sig:
            index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                metadata = pickle.load(f)
            return index, metadata

    # Rebuild index from knowledge_base/ documents
    metadata: List[Dict[str, str]] = []
    for file_path in sorted(KB_DIR.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            raw_text = extract_text_from_file(file_path)
            if not raw_text.strip():
                continue
            for chunk in chunk_document(raw_text, file_path.suffix.lower()):
                metadata.append({"source": file_path.name, "text": chunk})

    embedder = get_embedder()
    if not metadata:
        empty_index = faiss.IndexFlatIP(384)
        return empty_index, []

    texts = [item["text"] for item in metadata]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product on normalized vectors = Cosine Similarity
    index.add(embeddings)

    faiss.write_index(index, str(index_file))
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
    sig_file.write_text(current_sig, encoding="utf-8")

    return index, metadata


def search_kb(query: str) -> str:
    """Performs Hybrid Search (FAISS Vector Cosine Similarity + Keyword Boost) across all files in knowledge_base/."""
    if not KB_DIR.exists():
        return "NO_MATCHING_DOCUMENT_FOUND: Knowledge base directory does not exist."

    index, metadata = build_or_load_faiss_index()
    if not metadata or index.ntotal == 0:
        return "NO_MATCHING_DOCUMENT_FOUND: Knowledge base is empty."

    # 1. Semantic Search via FAISS
    embedder = get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    top_k = min(5, len(metadata))
    distances, indices = index.search(query_vec, top_k)

    # 2. Keyword Score Calculation for Hybrid Ranking
    stop_words = {
        "the", "and", "for", "with", "how", "can", "get", "what", "where",
        "was", "were", "want", "need", "from", "this", "that", "have", "has",
        "you", "your", "are", "not", "but", "all", "any", "do", "does"
    }
    query_tokens = [
        w for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        if w not in stop_words
    ]
    synonyms = {
        "late": ["delayed", "delivery", "shipping", "arrive", "order", "timelines"],
        "arrive": ["delivery", "shipping", "late", "delayed"],
        "return": ["refund", "window", "money", "shoes"],
        "broken": ["damaged", "replacement", "defective", "blender", "warranty"],
        "password": ["login", "reset", "account", "forgot", "sign"],
        "charged": ["billing", "duplicate", "card", "payment", "disputes"],
    }
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        if token in synonyms:
            expanded_tokens.update(synonyms[token])

    hybrid_results = []
    for rank_idx, doc_idx in enumerate(indices[0]):
        if doc_idx < 0 or doc_idx >= len(metadata):
            continue
        semantic_score = float(distances[0][rank_idx])
        item = metadata[doc_idx]
        text_lower = item["text"].lower()

        keyword_hits = sum(2 if t in query_tokens else 1 for t in expanded_tokens if t in text_lower)
        keyword_boost = min(0.25, keyword_hits * 0.05)
        hybrid_score = semantic_score + keyword_boost

        # Strict relevance gate so out-of-scope queries (e.g., 500 custom gift cards) return NO_MATCHING_DOCUMENT_FOUND
        if hybrid_score >= 0.42 and (semantic_score >= 0.35 or keyword_hits >= 2):
            hybrid_results.append((hybrid_score, item["source"], item["text"]))

    hybrid_results.sort(key=lambda x: x[0], reverse=True)

    if not hybrid_results:
        return "NO_MATCHING_DOCUMENT_FOUND: No relevant policy or FAQ passages exist in the company knowledge base for this request."

    top_passages = hybrid_results[:3]
    formatted_output = []
    for score, filename, text in top_passages:
        formatted_output.append(f"[Source File: {filename} | Relevance Score: {score:.2f}]\n{text}")

    return "\n\n---\n\n".join(formatted_output)


def create_knowledge_agent(model_client) -> AssistantAgent:
    system_message = """You are the Knowledge Base Agent.
Your ONLY job is to search the company's official documents using the `search_kb` hybrid FAISS vector search tool based on the Customer Query Agent's summary and category.

Instructions:
1. Call `search_kb` with a natural language search query describing the customer's core problem.
2. Output the exact matching passages returned by the tool along with their exact source file names (e.g., `shipping_policy.md`).
3. If the tool returns `NO_MATCHING_DOCUMENT_FOUND` or the passages do not address the customer's specific request (e.g., custom B2B gift cards for 500 employees), explicitly output:
   - Matched_Sources: None
   - Passages: NO_MATCHING_DOCUMENT_FOUND
Do NOT write the final reply to the customer."""

    return AssistantAgent(
        name="Knowledge_Base_Agent",
        description="Searches company documents (.md, .txt, .pdf, .docx, .csv, .json) using FAISS vector embeddings + keyword hybrid search and returns matching passages with source file names.",
        model_client=model_client,
        tools=[search_kb],
        system_message=system_message,
    )