"""Chunk-level reranking and token-level scoring.

Provides cross-encoder reranking of chunk lines and optional token-level
embedding scores for lightweight/static embedders (Model2Vec).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


def split_lines(text: str) -> List[str]:
    """Split chunk text into non-empty lines."""
    return [line for line in text.splitlines() if line.strip()]


_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def translation_tokens(line: str) -> List[str]:
    r"""Extract word-tokens from the translation part (after the first ``:``).

    Lines without a colon (headword-only, no translation) yield ``[]``.
    Tokens use ``[^\W\d_]+`` with ``re.UNICODE`` to keep macron vowels.
    """
    idx = line.find(":")
    if idx < 0:
        return []
    tail = line[idx + 1 :]
    return _TOKEN_RE.findall(tail)


class Reranker(Protocol):
    """Minimal reranker protocol (matches :class:`backends.Reranker`)."""

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]: ...


def rank_lines(
    reranker: Reranker, context: str, lines: List[str]
) -> List[Dict[str, Any]]:
    """Cross-encoder rerank *lines* against *context*.

    Returns results in **original line order** with fields::

        {"index": int, "text": str, "score": float, "rank": int}

    ``rank`` 0 = most relevant.
    """
    if not lines:
        return []

    ranked = reranker.rerank(context, lines, top_n=len(lines))

    # ranked is sorted by score desc; assign rank 0..N-1
    by_index: Dict[int, Dict[str, Any]] = {}
    for rank, item in enumerate(ranked):
        idx = item["index"]
        by_index[idx] = {
            "index": idx,
            "text": lines[idx],
            "score": float(item["relevance_score"]),
            "rank": rank,
        }

    # Fill any lines the reranker might have skipped
    result: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if i in by_index:
            result.append(by_index[i])
        else:
            result.append({"index": i, "text": line, "score": 0.0, "rank": len(ranked)})
    return result


class Embedder(Protocol):
    """Minimal embedder protocol (matches :class:`backends.Embedder`)."""

    @property
    def dim(self) -> int: ...

    def get_embeddings(self, texts: List[str]) -> np.ndarray: ...


def token_scores(
    embedder: Embedder,
    query: str,
    lines: List[str],
    *,
    min_score: Optional[float] = None,
) -> List[List[Dict[str, Any]]]:
    """Token-level cosine scores between query and translation tokens.

    All unique tokens (query + translation tokens across all lines) are
    embedded once in a single batch.  For each translation token in a line
    the maximum cosine similarity over the query tokens is returned.

    Args:
        embedder: Embedder with ``get_embeddings``.
        query: The search query string.
        lines: Chunk lines (as produced by :func:`split_lines`).
        min_score: If set, tokens below this score are dropped from output.

    Returns:
        One list per line; each element is
        ``{"token": str, "score": float, "query_token": str}``.
    """
    if not lines:
        return []

    q_tokens = _TOKEN_RE.findall(query)
    all_line_tokens = [translation_tokens(line) for line in lines]

    # Collect unique tokens for batch embedding
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for toks in all_line_tokens:
        for t in toks:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)
    for t in q_tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    if not unique_tokens:
        return [[] for _ in lines]

    all_vecs = embedder.get_embeddings(unique_tokens)
    tok2vec = {t: all_vecs[i] for i, t in enumerate(unique_tokens)}

    q_vecs = np.array([tok2vec[t] for t in q_tokens], dtype=np.float32) if q_tokens else np.zeros((0, all_vecs.shape[1]), dtype=np.float32)

    results: List[List[Dict[str, Any]]] = []
    for toks in all_line_tokens:
        line_scores: List[Dict[str, Any]] = []
        for t in toks:
            t_vec = tok2vec[t]
            if q_vecs.shape[0] > 0:
                sims = q_vecs @ t_vec
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                best_q = q_tokens[best_idx]
            else:
                best_score = 0.0
                best_q = ""
            if min_score is None or best_score >= min_score:
                line_scores.append({"token": t, "score": best_score, "query_token": best_q})
        results.append(line_scores)
    return results


def annotate_chunk(
    chunk: Dict[str, Any],
    context: str,
    reranker: Reranker,
    *,
    embedder: Optional[Embedder] = None,
    min_token_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Annotate a chunk with ranked lines and optional token scores.

    Returns a **copy** of the chunk with added keys:

    - ``lines``: list of ``{"index", "text", "score", "rank"}`` (from
      :func:`rank_lines`)
    - ``best_line``: the line text with rank 0
    - ``tokens``: (only when *embedder* is given) list of per-line token
      scores from :func:`token_scores`
    """
    chunk = dict(chunk)
    lines = split_lines(chunk.get("text", ""))
    ranked = rank_lines(reranker, context, lines)
    chunk["lines"] = ranked
    best = min(ranked, key=lambda r: r["rank"]) if ranked else None
    chunk["best_line"] = best["text"] if best else ""

    if embedder is not None:
        chunk["tokens"] = token_scores(
            embedder, context, lines, min_score=min_token_score
        )
    return chunk
