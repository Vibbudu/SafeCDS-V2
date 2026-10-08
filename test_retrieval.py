"""Test script for Hybrid Clinical Retriever (FR2)."""

import sys
from src.retrieval.hybrid_retriever import HybridClinicalRetriever


def test_retriever():
    print("[*] Initializing HybridClinicalRetriever...")
    retriever = HybridClinicalRetriever(guidelines_path="data/guidelines.json", rrf_k=60)

    query = "Chronic Kidney Disease Stage 4 5 and Type 2 Diabetes Metformin"
    print(f"[*] Testing retrieval for query: '{query}'")

    results = retriever.retrieve(query, top_k=2)
    assert len(results) > 0, "Expected at least 1 retrieved guideline result"

    print(f"\n[+] Successfully retrieved {len(results)} guidelines via Hybrid RRF:")
    for i, res in enumerate(results, 1):
        print(f"  {i}. [{res['id']}] RRF Score: {res.get('rrf_score')} | {res['topic']}")
        print(f"     Excerpt: {res['text'][:120]}...")

    evidence = retriever.get_evidence_string(query, top_k=2)
    assert len(evidence) > 0, "Expected non-empty evidence string"
    print("\n[+] Formatted Evidence String Sample:")
    print(evidence[:250] + "...\n")
    print("[OK] All retrieval tests passed!")


if __name__ == "__main__":
    test_retriever()
