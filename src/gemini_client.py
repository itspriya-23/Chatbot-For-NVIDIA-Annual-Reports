"""
Gemini API Client for RAG operations.
Uses official Google AI Studio endpoints via robust HTTPS REST requests.
Rate-limit safe: Respects free-tier TPM and RPM quotas with small batches and adaptive backoff.
"""

import os
import time
import requests
import certifi
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBED_MODEL = "gemini-embedding-2"
FALLBACK_EMBED_MODEL = "gemini-embedding-2-preview"
GEN_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEN_MODEL = "gemini-3.8-flash"

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file or pass it to GeminiClient."
            )
        self.ca_cert = certifi.where()

    def get_embedding(self, text: str, max_retries: int = 5) -> List[float]:
        """Generate a 3072-dimensional embedding vector for a single query."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={self.api_key}"
        payload = {
            "content": {
                "parts": [{"text": text[:6000]}]
            }
        }

        for attempt in range(max_retries):
            try:
                res = requests.post(url, json=payload, verify=self.ca_cert, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("embedding", {}).get("values", [])
                elif res.status_code == 429:
                    wait_time = 8 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)
        return [0.0] * 3072

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 15, max_retries: int = 5) -> List[List[float]]:
        """
        Embed chunks in smaller 15-item batches to stay comfortably within free tier TPM limits.
        """
        all_embeddings: List[List[float]] = []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents?key={self.api_key}"

        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"  [Embeddings] Total Chunks: {len(texts)} | Batch Size: {batch_size} | Total Batches: {total_batches}", flush=True)

        for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
            chunk_batch = texts[i:i + batch_size]
            requests_payload = [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": t[:4000]}]}
                }
                for t in chunk_batch
            ]

            success = False
            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json={"requests": requests_payload}, verify=self.ca_cert, timeout=25)
                    if res.status_code == 200:
                        data = res.json()
                        embeddings_list = data.get("embeddings", [])
                        for item in embeddings_list:
                            vec = item.get("values", [])
                            all_embeddings.append(vec if len(vec) == 3072 else [0.0] * 3072)
                        success = True
                        print(f"  [Batch {batch_num}/{total_batches}] Embedded {len(chunk_batch)} chunks (Total: {len(all_embeddings)}/{len(texts)})", flush=True)
                        break
                    elif res.status_code == 429:
                        wait = 8 * (attempt + 1)
                        print(f"  [Batch {batch_num}] 429 rate limit ({res.text[:80]}). Waiting {wait}s...", flush=True)
                        time.sleep(wait)
                    else:
                        time.sleep(2)
                except Exception as e:
                    time.sleep(3)

            if not success:
                print(f"  [Warning] Batch {batch_num} fallback to zeros", flush=True)
                for _ in chunk_batch:
                    all_embeddings.append([0.0] * 3072)

            # Polite spacing between batches
            if batch_num < total_batches:
                time.sleep(2.5)

        return all_embeddings

    def generate(self, prompt: str, system_instruction: Optional[str] = None, max_retries: int = 3) -> str:
        """
        Generate grounded text using Gemini.
        Temperature=0.0 enforces deterministic, factual answers strictly bound to evidence.
        """
        models_to_try = [GEN_MODEL, FALLBACK_GEN_MODEL]

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 1024
                }
            }

            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json=payload, verify=self.ca_cert, timeout=25)
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "").strip()
                    elif res.status_code == 429:
                        time.sleep(4 * (attempt + 1))
                    elif res.status_code in (500, 503):
                        time.sleep(1.5)
                        break
                except Exception as e:
                    time.sleep(1)

        return "I apologize, but I encountered an error communicating with the language model. Please try again."
