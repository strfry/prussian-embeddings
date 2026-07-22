"""Tests for chunk reranking and token scoring (chunk_rerank.py)."""

import pytest
import numpy as np

from prussian_embeddings.chunk_rerank import (
    annotate_chunk,
    rank_lines,
    split_lines,
    token_scores,
    translation_tokens,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

class FakeReranker:
    """Protocol-conform reranker with deterministic relevance_scores."""

    def __init__(self, scores: dict[int, float] | None = None):
        self.scores = scores or {}

    def rerank(self, query, documents, top_n=10):
        items = []
        for i, doc in enumerate(documents):
            score = self.scores.get(i, float(len(doc)))
            items.append({"index": i, "relevance_score": score})
        items.sort(key=lambda x: x["relevance_score"], reverse=True)
        return items[:top_n]


# Word → one-hot vector (dim = len(VOCAB))
VOCAB = ["abendstern", "stern", "betain", "to", "star", "evening", "prayer", "pray"]
V2I = {w: i for i, w in enumerate(VOCAB)}


class FakeEmbedder:
    """One-hot embedder over a known vocabulary."""

    dim = len(VOCAB)

    def get_embeddings(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            low = t.lower()
            if low in V2I:
                v[V2I[low]] = 1.0
            vecs.append(v)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-10, a_max=None)
        return vecs / norms if False else np.array(vecs, dtype=np.float32)


# ── split_lines ──────────────────────────────────────────────────────────────

def test_split_lines_basic():
    assert split_lines("a\nb\nc") == ["a", "b", "c"]


def test_split_lines_skips_empty():
    assert split_lines("a\n\n\nb") == ["a", "b"]


def test_split_lines_whitespace_only_skipped():
    assert split_lines("a\n   \nb") == ["a", "b"]


def test_split_lines_empty_string():
    assert split_lines("") == []


# ── translation_tokens ───────────────────────────────────────────────────────

def test_translation_tokens_with_colon():
    result = translation_tokens("bētan (verb): to pray")
    assert "to" in result
    assert "pray" in result


def test_translation_tokens_headword_only():
    assert translation_tokens("arktisks (adjective)") == []


def test_translation_tokens_empty_tail():
    assert translation_tokens("word:") == []


def test_translation_tokens_preserves_macrons():
    tokens = translation_tokens("test: māte")
    assert "māte" in tokens


# ── rank_lines ───────────────────────────────────────────────────────────────

def test_rank_lines_original_order():
    lines = ["first", "second", "third"]
    reranker = FakeReranker({0: 1.0, 1: 3.0, 2: 2.0})
    result = rank_lines(reranker, "ctx", lines)
    # Result is in original order
    assert [r["index"] for r in result] == [0, 1, 2]
    assert result[1]["rank"] == 0  # second line is most relevant
    assert result[1]["score"] == 3.0


def test_rank_lines_best_line():
    lines = ["a", "b", "c"]
    reranker = FakeReranker({0: 0.5, 1: 9.0, 2: 1.0})
    result = rank_lines(reranker, "ctx", lines)
    best = min(result, key=lambda r: r["rank"])
    assert best["text"] == "b"


def test_rank_lines_empty():
    assert rank_lines(FakeReranker(), "ctx", []) == []


# ── annotate_chunk ───────────────────────────────────────────────────────────

def test_annotate_chunk_best_line():
    chunk = {"text": "line A\nline B\nline C", "lemma": "test"}
    reranker = FakeReranker({0: 0.1, 1: 9.0, 2: 0.5})
    result = annotate_chunk(chunk, "ctx", reranker)
    assert result["best_line"] == "line B"
    assert result["lines"][1]["rank"] == 0
    # Original chunk is not mutated
    assert "lines" not in chunk


def test_annotate_chunk_empty():
    chunk = {"text": "", "lemma": "empty"}
    result = annotate_chunk(chunk, "ctx", FakeReranker())
    assert result["best_line"] == ""
    assert result["lines"] == []


# ── token_scores ─────────────────────────────────────────────────────────────

def test_token_scores_identical_token():
    """Token in translation that also appears in query → score 1.0."""
    lines = ["word: stern"]
    embedder = FakeEmbedder()
    result = token_scores(embedder, "stern", lines)
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0]["token"] == "stern"
    assert result[0][0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_token_scores_headword_not_scored():
    """Headword (part before :) is never token-scored."""
    lines = ["bētan (verb): to pray"]
    embedder = FakeEmbedder()
    result = token_scores(embedder, "bētan", lines)
    tokens_scored = [r["token"] for r in result[0]]
    assert "bētan" not in tokens_scored


def test_token_scores_line_without_colon():
    """A line with no colon → no tokens at all."""
    lines = ["arktisks (adjective)"]
    embedder = FakeEmbedder()
    result = token_scores(embedder, "arctic", lines)
    assert result[0] == []


def test_token_scores_min_score_filter():
    lines = ["word: to pray"]
    embedder = FakeEmbedder()
    # With min_score=1.0 only exact matches survive
    result_high = token_scores(embedder, "pray", lines, min_score=1.0)
    tokens_high = [r["token"] for r in result_high[0]]
    assert "pray" in tokens_high

    # With min_score=0.0 everything shows up
    result_all = token_scores(embedder, "pray", lines, min_score=0.0)
    assert len(result_all[0]) >= 1


def test_token_scores_empty_lines():
    assert token_scores(FakeEmbedder(), "query", []) == []
