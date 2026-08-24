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


# Two extra token streams over the same documents.  They exist because exact
# tokens alone have a blind spot the dense channel does not cover: the agent
# queries with inflected forms ("ženklai" for "ženklas", "senovės" for
# "senovė"), and BM25 has no stemmer, so it returns *nothing at all* for 99% of
# those.  Measured on 250 inflected Lithuanian queries, recall@1 went
# 0.004 (exact tokens) / 0.388 (dense) -> 0.764 with these two added.
# Neither needs a re-embedding: they are built from the store text at load.

PREFIX_LEN = 5


def prefix_tokens(text: str, n: int = PREFIX_LEN) -> List[str]:
    """Tokens truncated to their first *n* characters — a stemmer-free way to
    let a Baltic inflected form still match its lemma."""
    return [t[:n] for t in tokenize(text)]


NGRAM_LEN = 4


def ngram_tokens(text: str, n: int = NGRAM_LEN) -> List[str]:
    """Character *n*-grams per token, boundary-marked (``^word$``) so prefixes
    and suffixes stay distinguishable.  Catches stem changes that prefix
    truncation misses."""
    out: List[str] = []
    for t in tokenize(text):
        s = f"^{t}$"
        out.extend(s[i : i + n] for i in range(max(1, len(s) - n + 1)))
    return out


class BM25Index:
    """Self-contained Okapi BM25 index (no external dependencies).

    Parameters
    ----------
    texts : list[str]
        Documents to index.
    k1, b : float
        BM25 tuning parameters (default 1.5 / 0.75).
    """

    def __init__(
        self,
        texts: List[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or tokenize
        self.texts = texts
        self.N = len(texts)
        self.avgdl = 0.0
        self.doc_lens: List[int] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}

        total_len = 0
        for text in texts:
            tokens = self.tokenizer(text)
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
        tokens = self.tokenizer(text)
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


def weighted_rrf(
    rankings: List[Tuple[List[Tuple[Any, float]], float]],
    *,
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """RRF with a per-channel weight.

    Plain RRF gives every channel's rank-1 the same 1/(k+1), which is wrong
    whenever the channels differ in precision.  Measured on 250 literal
    Lithuanian gloss queries: exact-token BM25 alone hits recall@1 0.980,
    dense alone 0.500 — and fusing them 1:1 *lowered* the result to 0.924,
    because a confident lexical hit was routinely outranked by a dense
    look-alike.  Weighting restores it (0.988) without losing the dense
    channel's coverage where lexical matching finds nothing.
    """
    fused: Dict[Any, float] = {}
    for ranking, weight in rankings:
        for rank, item in enumerate(ranking):
            doc_id = item[0]
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


# Channel weights for :func:`hybrid_query`.  Exact tokens are the most precise
# signal and lead; prefix and n-gram cover inflection; dense covers the rest.
W_EXACT = 3.0
W_PREFIX = 1.5
W_NGRAM = 1.0
W_DENSE = 1.0


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
    bm25_prefix: Optional[BM25Index] = None,
    bm25_ngram: Optional[BM25Index] = None,
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
    bm25_prefix, bm25_ngram : BM25Index, optional
        Lexical channels over :func:`prefix_tokens` / :func:`ngram_tokens` of
        the same documents.  Supply both to cover inflected queries; without
        them this degrades to exact-token BM25 + dense (weighted).

    Returns
    -------
    list[(record, rrf_score)]
        Merged results, best first, limited to *k*.
    """
    # Dense candidates — use top_k directly to preserve store doc indices
    query_text = query_prefix + text if query_prefix else text
    query_vec = embedder.get_embedding(query_text, is_query=True)
    dense_top = store.top_k(query_vec, k=candidates)

    # BM25 candidates — already returns (doc_idx, score)
    bm25_results = bm25.query(text, k=candidates)

    channels: List[Tuple[List[Tuple[Any, float]], float]] = [
        (bm25_results, W_EXACT),
        (dense_top, W_DENSE),
    ]
    if bm25_prefix is not None:
        channels.append((bm25_prefix.query(text, k=candidates), W_PREFIX))
    if bm25_ngram is not None:
        channels.append((bm25_ngram.query(text, k=candidates), W_NGRAM))

    fused = weighted_rrf(channels, k=rrf_k)

    # Map doc_idx → record
    results: List[Tuple[Dict[str, Any], float]] = []
    for idx, score in fused[:k]:
        if 0 <= idx < len(store.records):
            results.append((store.records[idx], score))
    return results
