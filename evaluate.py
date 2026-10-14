"""
Evaluation Benchmark for NVIDIA Financial RAG Chatbot.
Evaluates answer accuracy, retrieval relevance, citation presence, and appropriate abstention
against the curated golden dataset.
"""

import os
import sys
import json
import time

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rag_engine import RAGEngine

DATASET_PATH = os.path.join(ROOT_DIR, "eval", "golden_dataset.json")
REPORT_PATH = os.path.join(ROOT_DIR, "eval", "evaluation_report.json")

def evaluate_system():
    print("=" * 75)
    print("NVIDIA RAG PIPELINE: AUTOMATED EVALUATION BENCHMARK")
    print("=" * 75)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Golden dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    engine = RAGEngine()
    if not engine.is_loaded:
        print("Error: Vector store is not ready. Please complete ingestion first.")
        return

    results = []
    total_cases = len(test_cases)
    correct_count = 0
    citation_count = 0
    abstention_tested = 0
    abstention_correct = 0

    print(f"Running evaluation on {total_cases} test cases across multiple categories...\n")

    for idx, case in enumerate(test_cases, start=1):
        q_id = case["id"]
        question = case["question"]
        category = case["category"]
        expected_keywords = case.get("expected_keywords", [])
        expected_doc = case.get("expected_document", "None")
        should_abstain = case.get("should_abstain", False)

        print(f"[{idx}/{total_cases}] Test: {q_id} ({category})")
        print(f"  Q: {question}")

        t0 = time.time()
        res = engine.query(question, top_k=5)
        latency = round(time.time() - t0, 2)

        answer = res["answer"]
        sources = res["sources"]
        abstained = res["abstained"]
        top_score = res.get("top_score", 0.0)

        # 1. Check Abstention correctness
        if should_abstain:
            abstention_tested += 1
            is_abstention_correct = abstained
            if is_abstention_correct:
                abstention_correct += 1
                is_correct = True
            else:
                is_correct = False
        else:
            # 2. Check Factual / Keyword correctness
            is_correct = any(kw.lower() in answer.lower() for kw in expected_keywords)

        # 3. Check Citation Presence (e.g. "[... Page ...]" or document citations)
        has_citations = any(c in answer for c in ["[", "Page", "FY20", "FY19", "Annual Report", "Proxy"])

        if is_correct:
            correct_count += 1
        if has_citations:
            citation_count += 1

        status = "PASSED" if is_correct else "FAILED"
        print(f"  Result: {status} | Latency: {latency}s | Top Similarity: {top_score:.3f}")
        print(f"  Has Citations: {has_citations} | Abstained: {abstained}")
        print(f"  Answer Snippet: {answer[:180].replace(chr(10), ' ')}...\n")

        results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "latency_seconds": latency,
            "top_similarity_score": top_score,
            "should_abstain": should_abstain,
            "abstained": abstained,
            "passed": is_correct,
            "has_citations": has_citations,
            "answer": answer,
            "retrieved_sources": [s["citation"] for s in sources[:3]]
        })

    # Summary Metrics
    accuracy_rate = (correct_count / total_cases) * 100
    citation_rate = (citation_count / total_cases) * 100
    abstention_accuracy = (abstention_correct / abstention_tested * 100) if abstention_tested > 0 else 100.0

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_cases,
        "passed_cases": correct_count,
        "failed_cases": total_cases - correct_count,
        "accuracy_rate_percent": round(accuracy_rate, 1),
        "citation_presence_percent": round(citation_rate, 1),
        "abstention_accuracy_percent": round(abstention_accuracy, 1),
        "test_results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 75)
    print("EVALUATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Total Test Cases Evaluated : {total_cases}")
    print(f"Overall Accuracy Rate      : {accuracy_rate:.1f}% ({correct_count}/{total_cases})")
    print(f"Citation Verification Rate : {citation_rate:.1f}%")
    print(f"Appropriate Abstention Rate: {abstention_accuracy:.1f}% ({abstention_correct}/{abstention_tested})")
    print(f"Full Report Saved To       : {os.path.relpath(REPORT_PATH)}")
    print("=" * 75)

if __name__ == "__main__":
    evaluate_system()

