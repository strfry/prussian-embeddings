"""Hybrid search: BM25 + dense retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_TOKENIZER_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase ``\\w+``-Unicode tokens (macrons preserved via Unicode)."""
    return [t.lower() for t in _TOKENIZER_RE.findall(text)]


class BM25Index:
    """Self-contained Okapi BM25 index (no external dependencies).

    Parameters
    ----------
    texts : list[str]
        Documents to index.
    k1, b : float
        BM25 tuning parameters (default 1.5 / 0.75).
    """

    def __init__(self, texts: List[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.texts = texts
        self.N = len(texts)
        self.avgdl = 0.0
        self.doc_lens: List[int] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}

        total_len = 0
        for text in texts:
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.tf.append(freq)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = total_len / max(1, self.N)

    def query(self, text: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return ``[(doc_idx, score)]`` for the top-*k* documents (scores > 0)."""
        tokens = tokenize(text)
        scores: Dict[int, float] = {}
        for t in tokens:
            if t not in self.df:
                continue
            df = self.df[t]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i in range(self.N):
                tf_val = self.tf[i].get(t, 0)
                if tf_val == 0:
                    continue
                dl = self.doc_lens[i]
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
                scores[i] = scores.get(i, 0.0) + idf * numerator / denominator
        ranked = sorted(
            ((idx, s) for idx, s in scores.items() if s > 0),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:k]


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[Any, float]]],
    *,
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """Reciprocal Rank Fusion over an arbitrary number of rankings.

    Each ranking is ``[(id, …)]`` ordered best-first.  Ties are broken
    deterministically by score (desc) then id.
    """
    fused: Dict[Any, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            doc_id = item[0]
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    result = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return result


def hybrid_query(
    store: Any,
    embedder: Any,
    bm25: BM25Index,
    text: str,
    k: int = 10,
    *,
    query_prefix: str = "",
    candidates: int = 50,
    rrf_k: int = 60,
) -> List[Tuple[Dict[str, Any], float]]:
    """Hybrid dense + BM25 retrieval with RRF fusion.

    Parameters
    ----------
    store : EmbeddingStore
        Dense embedding store with ``top_k(query_vec, k)`` and ``records``.
    embedder : Embedder
        For dense query embedding.
    bm25 : BM25Index
        Over the same record texts.
    text : str
        User query.
    k : int
        Final number of results.
    query_prefix : str
        Prefix for the dense query (e.g. ``"query: "`` for e5).
    candidates : int
        Number of candidates drawn from each retrieval channel.
    rrf_k : int
        RRF ``k`` parameter (default 60).

    Returns
    -------
    list[(record, rrf_score)]
        Merged results, best first, limited to *k*.
    """
    # Dense candidates — use top_k directly to preserve store doc indices
    query_text = query_prefix + text if query_prefix else text
    query_vec = embedder.get_embedding(query_text)
    dense_top = store.top_k(query_vec, k=candidates)

    # BM25 candidates — already returns (doc_idx, score)
    bm25_results = bm25.query(text, k=candidates)

    # RRF fusion over unified store document indices
    fused = reciprocal_rank_fusion([dense_top, bm25_results], k=rrf_k)

    # Map doc_idx → record
    results: List[Tuple[Dict[str, Any], float]] = []
    for idx, score in fused[:k]:
        if 0 <= idx < len(store.records):
            results.append((store.records[idx], score))
    return results
