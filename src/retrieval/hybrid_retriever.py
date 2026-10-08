"""Hybrid Dense + Sparse Clinical Guideline Retriever with Reciprocal Rank Fusion (RRF)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("safecds.retrieval")


class HybridClinicalRetriever:
    """Combines dense semantic embeddings (FAISS + MiniLM) with sparse lexical search (BM25)

    fused via Reciprocal Rank Fusion (k=60).
    """

    def __init__(self, guidelines_path: str = "data/guidelines.json", rrf_k: int = 60):
        self.guidelines_path = Path(guidelines_path)
        self.rrf_k = rrf_k
        self.documents: List[Dict[str, Any]] = []
        self.corpus_texts: List[str] = []

        # Sparse components
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []

        # Dense components
        self.embedder = None
        self.faiss_index = None

        self._load_corpus()
        self._init_sparse()
        self._init_dense()

    def _load_corpus(self) -> None:
        """Load curated guideline passages from JSON."""
        if not self.guidelines_path.exists():
            raise FileNotFoundError(f"Guidelines file not found at: {self.guidelines_path}")

        with open(self.guidelines_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.corpus_texts = [
            f"{doc.get('topic', '')}: {doc.get('text', '')}"
            for doc in self.documents
        ]
        logger.info("Loaded %d guideline passages from %s", len(self.documents), self.guidelines_path)

    def _init_sparse(self) -> None:
        """Initialize BM25 index over tokenized corpus."""
        try:
            from rank_bm25 import BM25Okapi

            self.tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info("BM25 sparse index initialized successfully.")
        except ImportError:
            logger.warning("rank_bm25 not installed. Sparse retrieval falling back to keyword overlap.")
            self.bm25 = None

    def _init_dense(self) -> None:
        """Initialize sentence embeddings and FAISS inner product index."""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self.embedder.encode(self.corpus_texts, convert_to_numpy=True)
            # L2 normalize embeddings for cosine similarity via inner product
            faiss.normalize_L2(embeddings)

            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
            logger.info("FAISS dense index initialized (dim=%d, count=%d).", dimension, self.faiss_index.ntotal)
        except Exception as e:
            logger.warning("Dense retriever initialization deferred: %s", e)
            self.embedder = None
            self.faiss_index = None

    def _sparse_search(self, query: str) -> List[int]:
        """Return document indices sorted by sparse score."""
        tokens = query.lower().split()
        if self.bm25:
            scores = self.bm25.get_scores(tokens)
            ranked_indices = list(range(len(self.documents)))
            ranked_indices.sort(key=lambda idx: scores[idx], reverse=True)
            return ranked_indices
        else:
            # Fallback simple keyword match
            scores = [
                sum(1 for token in tokens if token in text.lower())
                for text in self.corpus_texts
            ]
            ranked_indices = list(range(len(self.documents)))
            ranked_indices.sort(key=lambda idx: scores[idx], reverse=True)
            return ranked_indices

    def _dense_search(self, query: str) -> List[int]:
        """Return document indices sorted by dense embedding cosine similarity."""
        if not self.embedder or not self.faiss_index:
            return list(range(len(self.documents)))

        import faiss

        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        _, indices = self.faiss_index.search(query_vector, len(self.documents))
        return indices[0].tolist()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Execute hybrid search fusing dense and sparse ranks via Reciprocal Rank Fusion.

        RRF(d) = sum(1 / (k + rank(d)))
        """
        sparse_ranks = self._sparse_search(query)
        dense_ranks = self._dense_search(query)

        rrf_scores: Dict[int, float] = {i: 0.0 for i in range(len(self.documents))}

        for rank, doc_idx in enumerate(sparse_ranks, start=1):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank)

        for rank, doc_idx in enumerate(dense_ranks, start=1):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx in sorted_indices[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["rrf_score"] = round(rrf_scores[idx], 5)
            results.append(doc_copy)

        return results

    def get_evidence_string(self, query: str, top_k: int = 3) -> str:
        """Helper to format retrieved guidelines into a prompt-ready evidence string."""
        docs = self.retrieve(query, top_k=top_k)
        passages = [
            f"[{d.get('source', 'Guideline')}] {d.get('topic')}: \"{d.get('text')}\""
            for d in docs
        ]
        return "\n\n".join(passages)
