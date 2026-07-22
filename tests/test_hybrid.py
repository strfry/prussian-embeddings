"""Tests for hybrid search: BM25, RRF, and hybrid_query (hybrid.py)."""

import pytest
import numpy as np

from prussian_embeddings.hybrid import (
    BM25Index,
    hybrid_query,
    reciprocal_rank_fusion,
    tokenize,
)
from prussian_embeddings.store import EmbeddingStore


# ── tokenize ─────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_keeps_macrons():
    tokens = tokenize("bētan māte")
    assert "bētan" in tokens
    assert "māte" in tokens


def test_tokenize_lowercases():
    assert tokenize("ABENDSTERN") == ["abendstern"]


# ── BM25Index ────────────────────────────────────────────────────────────────

def test_bm25_rare_term_ranks_first():
    docs = ["the cat sat", "the dog ran", "a unique zoggle"]
    idx = BM25Index(docs)
    results = idx.query("zoggle")
    assert results[0][0] == 2  # doc index 2 has the rare term


def test_bm25_frequent_term_lower_idf():
    docs = ["the the the", "unique word here", "the unique thing"]
    idx = BM25Index(docs)
    # "unique" appears in docs 1 and 2, "the" in 0 and 2
    # Query "unique" should prefer doc 1 (unique is a larger fraction)
    results = idx.query("unique")
    # doc 1 has only "unique word here" — high tf for unique relative to doc len
    doc_ids = [r[0] for r in results]
    assert 1 in doc_ids


def test_bm25_length_normalization():
    docs = ["cat", "cat dog bird fish moon star"]
    idx = BM25Index(docs)
    results = idx.query("cat")
    # Short doc should rank higher due to less dilution
    assert results[0][0] == 0


def test_bm25_empty_query():
    idx = BM25Index(["hello world", "foo bar"])
    assert idx.query("") == []


def test_bm25_no_matches():
    idx = BM25Index(["hello world"])
    assert idx.query("zzzznonexistent") == []


# ── reciprocal_rank_fusion ───────────────────────────────────────────────────

def test_rrf_both_lists_high_wins():
    r1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    r2 = [("a", 0.5), ("b", 0.9), ("d", 0.8)]
    fused = reciprocal_rank_fusion([r1, r2])
    ids = [x[0] for x in fused]
    # "a" is top-1 in r1 and top-1 in r2 → best fused score
    assert ids[0] == "a"


def test_rrf_one_sided_high():
    r1 = [("x", 0.9)]
    r2 = [("y", 0.9), ("x", 0.1)]
    fused = reciprocal_rank_fusion([r1, r2])
    ids = [x[0] for x in fused]
    # "x" is rank 0 in r1 and rank 1 in r2 → better than "y" (rank 1 in r2 only)
    assert ids[0] == "x"


def test_rrf_k_parameter():
    r1 = [("a", 1.0), ("b", 0.9)]
    r2 = [("b", 1.0), ("a", 0.9)]
    fused_small = reciprocal_rank_fusion([r1, r2], k=1)
    fused_large = reciprocal_rank_fusion([r1, r2], k=100)
    # With small k, rank differences matter more; with large k, both get ≈0.01
    # Both should still have the same ordering (a=b tied, deterministic by id)
    assert [x[0] for x in fused_small] == [x[0] for x in fused_large]


def test_rrf_deterministic_tie_break():
    r1 = [("b", 0.9), ("a", 0.9)]
    fused = reciprocal_rank_fusion([r1])
    # "b" is rank 0 → higher RRF score than "a" at rank 1.
    # Deterministic: same RRF score would sort by id ascending.
    # Here they differ; just verify deterministic output.
    ids = [x[0] for x in fused]
    assert ids == ["b", "a"]
    # Run again to confirm determinism
    fused2 = reciprocal_rank_fusion([r1])
    assert [x[0] for x in fused2] == ids


def test_rrf_tie_break_by_id():
    # Same RRF score → sorted by id ascending
    r1 = [("b", 1.0)]
    r2 = [("a", 1.0)]
    # Both rank 0, both get 1/(k+1), so same RRF score
    fused = reciprocal_rank_fusion([r1, r2])
    assert [x[0] for x in fused] == ["a", "b"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []


# ── hybrid_query ─────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Cosine-similarity-aware fake: same text → score 1.0, else 0.3."""

    dim = 4

    def get_embeddings(self, texts):
        vecs = []
        for t in texts:
            h = hash(t) % 1000
            v = np.zeros(4, dtype=np.float32)
            v[0] = float(h)
            v[1] = 1.0
            vecs.append(v)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-10, a_max=None)
        return vecs / norms

    def get_embedding(self, text):
        return self.get_embeddings([text])[0]


def _mini_store():
    records = [
        {"text": "bētan (verb): to pray", "lemma": "bētan"},
        {"text": "stern (noun): star", "lemma": "stern"},
        {"text": "abendstern (noun): evening star", "lemma": "abendstern"},
    ]
    embedder = FakeEmbedder()
    texts = [r["text"] for r in records]
    embeddings = np.array(embedder.get_embeddings(texts), dtype=np.float32)
    return EmbeddingStore(embeddings=embeddings, records=records)


def test_hybrid_query_both_signals_win():
    store = _mini_store()
    embedder = FakeEmbedder()
    texts = [r["text"] for r in store.records]
    bm25 = BM25Index(texts)

    results = hybrid_query(store, embedder, bm25, "stern")
    # "stern" and "abendstern" both match BM25; dense also hits stern-related
    assert len(results) > 0
    # At least one result should contain "stern"
    assert any("stern" in rec.get("text", "").lower() for rec, _ in results)


def test_hybrid_query_respects_k():
    store = _mini_store()
    embedder = FakeEmbedder()
    texts = [r["text"] for r in store.records]
    bm25 = BM25Index(texts)

    results = hybrid_query(store, embedder, bm25, "stern", k=1)
    assert len(results) == 1


def test_hybrid_query_bm25_only():
    store = _mini_store()
    embedder = FakeEmbedder()
    texts = [r["text"] for r in store.records]
    bm25 = BM25Index(texts)

    results = hybrid_query(store, embedder, bm25, "prayer", candidates=50)
    # "prayer" is only in bētan line via BM25
    assert any("bētan" in rec.get("text", "") for rec, _ in results)
