"""
Document Ingestion Pipeline for NVIDIA Annual Reports and Proxy Statements.
Parses PDFs, extracts page-by-page text, chunks into semantic segments with citations,
generates Gemini embeddings using batchEmbedContents, and persists the vector database to disk.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import re
import json
import time
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from dotenv import load_dotenv

logging.getLogger("pypdf").setLevel(logging.ERROR)

from src.gemini_client import GeminiClient
from src.vector_store import SimpleVectorStore

load_dotenv()

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")

DOC_METADATA = {
    "NVIDIA_2026_Annual_Report.pdf": {"year": 2026, "type": "Annual Report", "title": "NVIDIA 2026 Annual Report (Form 10-K)"},
    "NVIDIA_2025_Annual_Report.pdf": {"year": 2025, "type": "Annual Report", "title": "NVIDIA 2025 Annual Report (Form 10-K)"},
    "NVIDIA_2024_Annual_Report.pdf": {"year": 2024, "type": "Annual Report", "title": "NVIDIA 2024 Annual Report (Form 10-K)"},
    "NVIDIA_2023_Annual_Report.pdf": {"year": 2023, "type": "Annual Report", "title": "NVIDIA 2023 Annual Report (Form 10-K)"},
    "NVIDIA_2020_Annual_Report.pdf": {"year": 2020, "type": "Annual Report", "title": "NVIDIA 2020 Annual Report (Form 10-K)"},
    "NVIDIA_2018_Annual_Report.pdf": {"year": 2018, "type": "Annual Report", "title": "NVIDIA 2018 Annual Report (Form 10-K)"},
    "NVIDIA_2015_Annual_Report.pdf": {"year": 2015, "type": "Annual Report", "title": "NVIDIA 2015 Annual Report (Form 10-K)"},
    "NVIDIA_2008_Proxy_Statement.pdf": {"year": 2008, "type": "Proxy Statement", "title": "NVIDIA 2008 Proxy Statement (DEF 14A)"},
    "NVIDIA_2004_Proxy_Statement.pdf": {"year": 2004, "type": "Proxy Statement", "title": "NVIDIA 2004 Proxy Statement (DEF 14A)"},
    "NVIDIA_1999_Proxy_Statement.pdf": {"year": 1999, "type": "Proxy Statement", "title": "NVIDIA 1999 Proxy Statement (DEF 14A)"},
    "NVIDIA_1998_Proxy_Statement.pdf": {"year": 1998, "type": "Proxy Statement", "title": "NVIDIA 1998 Proxy Statement (DEF 14A)"},
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunk = text[start:].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            break

        break_pos = -1
        p_break = text.rfind("\n\n", start + chunk_size // 2, end)
        if p_break != -1:
            break_pos = p_break + 2
        else:
            s_break = max(text.rfind(". ", start + chunk_size // 2, end),
                          text.rfind(";\n", start + chunk_size // 2, end))
            if s_break != -1:
                break_pos = s_break + 2
            else:
                n_break = text.rfind("\n", start + chunk_size // 2, end)
                if n_break != -1:
                    break_pos = n_break + 1
                else:
                    break_pos = end

        chunk = text[start:break_pos].strip()
        if len(chunk) > 50:
            chunks.append(chunk)

        start = max(start + 1, break_pos - chunk_overlap)

    return chunks

def extract_chunks_from_pdf(pdf_path: str, meta: Dict[str, Any], max_pages_per_doc: int = 15) -> List[Dict[str, Any]]:
    chunks = []
    filename = os.path.basename(pdf_path)
    title = meta.get("title", filename)
    year = meta.get("year", "Unknown")
    doc_type = meta.get("type", "Report")

    print(f"[Ingest] Scanning '{filename}' ({doc_type}, FY{year})...", flush=True)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_to_process = min(total_pages, max_pages_per_doc)

    chunk_idx = 0
    for page_idx in range(pages_to_process):
        page_num = page_idx + 1
        try:
            page = reader.pages[page_idx]
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)

            if len(cleaned) < 60:
                continue

            page_chunks = chunk_text(cleaned, chunk_size=1200, chunk_overlap=200)

            for p_chunk in page_chunks:
                chunk_idx += 1
                context_header = f"[Source: {title} | Year: FY{year} | Page: {page_num}]\n"
                full_text = context_header + p_chunk

                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_c{chunk_idx}",
                    "source_file": filename,
                    "title": title,
                    "year": year,
                    "doc_type": doc_type,
                    "page_number": page_num,
                    "citation": f"{title}, Page {page_num}",
                    "text": full_text,
                    "content_only": p_chunk
                })
        except Exception as e:
            print(f"  Warning: Failed page {page_num} of {filename}: {e}", flush=True)

    del reader
    print(f"  -> Extracted {len(chunks)} chunks from {filename} (scanned {pages_to_process} pages)", flush=True)
    return chunks

def run_ingestion(max_docs: int = None, max_pages_per_doc: int = 15):
    start_time = time.time()
    print("=" * 70, flush=True)
    print("NVIDIA RAG PIPELINE: DOCUMENT INGESTION & VECTOR INDEXING", flush=True)
    print("=" * 70, flush=True)

    client = GeminiClient()
    vector_store = SimpleVectorStore(storage_dir=VECTOR_STORE_DIR)

    all_chunks: List[Dict[str, Any]] = []

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pdf")]
    files.sort(reverse=True)

    if max_docs:
        files = files[:max_docs]

    print(f"Found {len(files)} filings in {RAW_DATA_DIR}", flush=True)

    for filename in files:
        pdf_path = os.path.join(RAW_DATA_DIR, filename)
        meta = DOC_METADATA.get(filename, {
            "year": int(re.search(r'\d{4}', filename).group(0)) if re.search(r'\d{4}', filename) else 2024,
            "type": "Annual Report" if "annual" in filename.lower() else "Proxy Statement",
            "title": filename.replace(".pdf", "").replace("_", " ")
        })
        chunks = extract_chunks_from_pdf(pdf_path, meta, max_pages_per_doc=max_pages_per_doc)
        all_chunks.extend(chunks)

    print(f"\n[Ingest] Total chunks extracted: {len(all_chunks)}", flush=True)
    print("[Ingest] Generating Gemini batch embeddings (15 chunks/batch)...", flush=True)

    chunk_texts = [c["text"] for c in all_chunks]
    t0 = time.time()
    embeddings = client.get_embeddings_batch(chunk_texts, batch_size=15)
    print(f"[Ingest] Generated {len(embeddings)} embeddings in {time.time()-t0:.1f}s!", flush=True)

    print("\n[Ingest] Persisting chunks and vectors to disk...", flush=True)
    vector_store.add_documents(all_chunks, embeddings)
    vector_store.save()

    elapsed = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"SUCCESS: Ingestion completed in {elapsed:.1f}s!", flush=True)
    print(f"Total Chunks Indexed: {len(all_chunks)}")
    print(f"Vector Store Directory: {os.path.abspath(VECTOR_STORE_DIR)}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_ingestion(max_docs=11, max_pages_per_doc=15)
