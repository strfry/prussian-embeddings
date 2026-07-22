"""Rank evaluation: ref-clustered chunks vs entry-baseline.

Proof that ref-clustered embedding improves findability of translation-less forms.

Compare chunk-store (entries grouped by FST lemma) vs baseline (entries in isolation).
Groups: A = has translations (control), B = no translations (test case).
Only clusters where at least one member has translations are considered.

For each query (Prussian word from an entry in a multi-member cluster):
- Chunk-store gold: the chunk index containing this entry
- Baseline-store gold: all baseline record indices whose word belongs to this cluster

Metrics: Hit@1/5/10, MRR per (store × group).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .backends import get_embedder
from .evaluate import Bucket, _l2_normalize
from .store import resolve_embedder_config
from .passages import has_translations
from .store import EmbeddingStore


@dataclass
class RefEvalQuery:
    """A single evaluation query."""

    word: str  # Prussian headword (the query)
    entry_idx: int  # Index of the entry (cluster member)
    cluster_lemma: str  # The cluster's lemma
    cluster_members: Set[str]  # All words in the cluster


def load_chunks_jsonl(path: str) -> Tuple[List[dict], Dict[int, Set[str]]]:
    """Load chunks from JSONL, return (chunks, lemma_to_members).

    Args:
        path: Path to embedding_chunks.jsonl

    Returns:
        (chunks, lemma_to_members) where lemma_to_members maps each cluster
        lemma to the set of member words.
    """
    chunks = []
    lemma_to_members = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            lemma = chunk.get("lemma", "")
            members = set(chunk.get("members", []))
            lemma_to_members[lemma] = members

    return chunks, lemma_to_members


def load_dictionary(path: str) -> Dict[str, dict]:
    """Load dictionary entries, keyed by word.

    Homographs (multiple entries per word): keep an entry with translations
    if any exists, so group A/B classification reflects the baseline store,
    which contains every translated homograph.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Expected list or dict in dictionary JSON")

    by_word: Dict[str, dict] = {}
    for e in entries:
        word = e.get("word", "")
        prev = by_word.get(word)
        if prev is None or (not has_translations(prev) and has_translations(e)):
            by_word[word] = e
    return by_word


def build_baseline_word_to_indices(baseline_store: EmbeddingStore) -> Dict[str, List[int]]:
    """Map baseline record words to their indices."""
    word_to_indices = defaultdict(list)
    for idx, record in enumerate(baseline_store.records):
        word = record.get("word", "")
        if word:
            word_to_indices[word].append(idx)
    return word_to_indices


def build_eval_queries(
    chunks: List[dict],
    dictionary: Dict[str, dict],
) -> Tuple[List[RefEvalQuery], int]:
    """Build evaluation queries from multi-member clusters with at least one translated member.

    Args:
        chunks: List of chunk dicts
        dictionary: Dictionary entries by word

    Returns:
        (queries, num_excluded_entries) tuple
    """
    queries = []
    excluded_count = 0

    for chunk_idx, chunk in enumerate(chunks):
        members = chunk.get("members", [])

        # Only multi-member clusters
        if len(members) <= 1:
            continue

        # Check if at least one member has translations
        has_translated = any(
            has_translations(dictionary.get(w, {})) for w in members
        )
        if not has_translated:
            excluded_count += len(members)
            continue

        # Add query for each cluster member
        for word in members:
            queries.append(
                RefEvalQuery(
                    word=word,
                    entry_idx=chunk_idx,
                    cluster_lemma=chunk.get("lemma", ""),
                    cluster_members=set(members),
                )
            )

    return queries, excluded_count


def split_queries_by_group(
    queries: List[RefEvalQuery], dictionary: Dict[str, dict]
) -> Tuple[List[RefEvalQuery], List[RefEvalQuery]]:
    """Split queries into group A (has translations) and B (no translations)."""
    group_a = []
    group_b = []

    for q in queries:
        entry = dictionary.get(q.word, {})
        if has_translations(entry):
            group_a.append(q)
        else:
            group_b.append(q)

    return group_a, group_b


def compute_chunk_gold_indices(
    query: RefEvalQuery, chunk_idx: int
) -> Set[int]:
    """Gold indices for chunk-store: just the chunk containing this entry."""
    return {chunk_idx}


def compute_baseline_gold_indices(
    query: RefEvalQuery,
    baseline_store: EmbeddingStore,
    word_to_baseline_indices: Dict[str, List[int]],
) -> Set[int]:
    """Gold indices for baseline-store: all records whose word is in the cluster."""
    gold_indices = set()
    for word in query.cluster_members:
        indices = word_to_baseline_indices.get(word, [])
        gold_indices.update(indices)
    return gold_indices


def evaluate_rank(
    queries: List[RefEvalQuery],
    embedder,
    chunk_store: EmbeddingStore,
    baseline_store: EmbeddingStore,
    word_to_baseline_indices: Dict[str, List[int]],
    device: str = "cuda",
    batch_size: int = 256,
) -> Tuple[Bucket, Bucket]:
    """Evaluate ranking for chunk-store and baseline-store.

    Returns:
        (chunk_bucket, baseline_bucket) both Bucket instances with Hit@k and MRR.
    """
    if not queries:
        return Bucket(), Bucket()

    chunk_meta = chunk_store.meta
    baseline_meta = baseline_store.meta
    chunk_prefix = chunk_meta.get("query_prefix", "")
    baseline_prefix = baseline_meta.get("query_prefix", "")

    # Check if store embeddings are normalized
    chunk_norms = np.linalg.norm(chunk_store.embeddings, axis=1)
    baseline_norms = np.linalg.norm(baseline_store.embeddings, axis=1)

    chunk_embeddings = chunk_store.embeddings
    baseline_embeddings = baseline_store.embeddings

    # Normalize if needed
    if not np.allclose(chunk_norms, 1.0, atol=0.01):
        chunk_embeddings = _l2_normalize(chunk_embeddings)
    if not np.allclose(baseline_norms, 1.0, atol=0.01):
        baseline_embeddings = _l2_normalize(baseline_embeddings)

    # Embed queries once per distinct store prefix
    def embed_queries(prefix: str) -> np.ndarray:
        texts = [prefix + q.word for q in queries]
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(embedder.get_embeddings(texts[i : i + batch_size], is_query=True))
        return _l2_normalize(np.vstack(vecs))

    chunk_q_vecs = embed_queries(chunk_prefix)
    baseline_q_vecs = (
        chunk_q_vecs if baseline_prefix == chunk_prefix else embed_queries(baseline_prefix)
    )

    # Compute scores
    chunk_scores = chunk_q_vecs @ chunk_embeddings.T
    chunk_scores = np.clip(chunk_scores, -1.0, 1.0)

    baseline_scores = baseline_q_vecs @ baseline_embeddings.T
    baseline_scores = np.clip(baseline_scores, -1.0, 1.0)

    # Buckets for chunk and baseline
    chunk_bucket = Bucket()
    baseline_bucket = Bucket()

    for qi, query in enumerate(queries):
        # Chunk-store ranking
        chunk_gold = compute_chunk_gold_indices(query, query.entry_idx)
        chunk_row = chunk_scores[qi]
        chunk_rank = _compute_any_rank(chunk_row, chunk_gold)

        # Baseline-store ranking
        baseline_gold = compute_baseline_gold_indices(
            query, baseline_store, word_to_baseline_indices
        )
        if baseline_gold:
            baseline_row = baseline_scores[qi]
            baseline_rank = _compute_any_rank(baseline_row, baseline_gold)
        else:
            baseline_rank = len(baseline_scores[0]) + 1  # No gold found

        # Update buckets
        for k, rank in [(1, chunk_rank), (5, chunk_rank), (10, chunk_rank)]:
            if rank <= k:
                if k == 1:
                    chunk_bucket.hits_1 += 1
                elif k == 5:
                    chunk_bucket.hits_5 += 1
                elif k == 10:
                    chunk_bucket.hits_10 += 1

        chunk_bucket.mrr_sum += 1.0 / chunk_rank
        chunk_bucket.n += 1

        for k, rank in [(1, baseline_rank), (5, baseline_rank), (10, baseline_rank)]:
            if rank <= k:
                if k == 1:
                    baseline_bucket.hits_1 += 1
                elif k == 5:
                    baseline_bucket.hits_5 += 1
                elif k == 10:
                    baseline_bucket.hits_10 += 1

        baseline_bucket.mrr_sum += 1.0 / baseline_rank
        baseline_bucket.n += 1

    return chunk_bucket, baseline_bucket


def _compute_any_rank(row: np.ndarray, gold_indices: Set[int]) -> int:
    """Compute best rank among all gold indices."""
    if not gold_indices:
        return len(row) + 1

    gold_indices = np.array(sorted(gold_indices))
    if len(gold_indices) > len(row):
        gold_indices = gold_indices[: len(row)]

    gold_scores = row[gold_indices]
    # Minimum number of scores strictly higher than any gold score
    rank = int(np.min(np.sum(row[:, None] > gold_scores[None, :], axis=0))) + 1
    return rank


def format_ref_report(results: Dict) -> str:
    """Format ref evaluation results as a fixed-width table."""
    lines = []
    lines.append("=== Ref-Clustered Chunk Evaluation ===")
    lines.append("")
    model = results.get("model")
    if model:
        lines.append(f"Model: {model}")

    excluded = results.get("num_excluded_entries", 0)
    if excluded > 0:
        lines.append(f"Excluded entries (clusters without translations): {excluded}")
    lines.append("")

    # Header
    header = (
        f"{'Store':<20} {'Group':<8} {'n':>6} {'Hit@1':>7} {'Hit@5':>7}"
        f" {'Hit@10':>7} {'MRR':>7}"
    )
    lines.append(header)

    # Chunk store
    chunk_a = results["chunk_bucket_a"]
    chunk_b = results["chunk_bucket_b"]

    if chunk_a.n > 0:
        d = chunk_a.as_dict()
        lines.append(
            f"{'chunk-store':<20} {'A (trans)':>8} {d['n']:>6} {d['Hit@1']:>7.3f}"
            f" {d['Hit@5']:>7.3f} {d['Hit@10']:>7.3f} {d['MRR']:>7.3f}"
        )

    if chunk_b.n > 0:
        d = chunk_b.as_dict()
        lines.append(
            f"{'chunk-store':<20} {'B (no tr)':>8} {d['n']:>6} {d['Hit@1']:>7.3f}"
            f" {d['Hit@5']:>7.3f} {d['Hit@10']:>7.3f} {d['MRR']:>7.3f}"
        )

    # Baseline store
    baseline_a = results["baseline_bucket_a"]
    baseline_b = results["baseline_bucket_b"]

    if baseline_a.n > 0:
        d = baseline_a.as_dict()
        lines.append(
            f"{'baseline-store':<20} {'A (trans)':>8} {d['n']:>6} {d['Hit@1']:>7.3f}"
            f" {d['Hit@5']:>7.3f} {d['Hit@10']:>7.3f} {d['MRR']:>7.3f}"
        )

    if baseline_b.n > 0:
        d = baseline_b.as_dict()
        lines.append(
            f"{'baseline-store':<20} {'B (no tr)':>8} {d['n']:>6} {d['Hit@1']:>7.3f}"
            f" {d['Hit@5']:>7.3f} {d['Hit@10']:>7.3f} {d['MRR']:>7.3f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Rank evaluation: ref-clustered chunks vs entry-baseline"
    )
    parser.add_argument(
        "--chunk-store",
        type=str,
        required=True,
        help="Path stem of chunk-store (e.g., data/embeddings_st_e5large_chunks)",
    )
    parser.add_argument(
        "--baseline-store",
        type=str,
        required=True,
        help="Path stem of baseline-store (e.g., data/embeddings_st_e5large)",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to embedding_chunks.jsonl",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="../fst/data/external/twanksta_entries.json",
        help="Path to dictionary JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedder (cuda, cpu, etc.)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to save full JSON results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding queries",
    )

    args = parser.parse_args()

    # Load chunks and dictionary
    chunks, _ = load_chunks_jsonl(args.chunks)
    dictionary = load_dictionary(args.dictionary)

    # Load stores
    chunk_store = EmbeddingStore.load(args.chunk_store)
    baseline_store = EmbeddingStore.load(args.baseline_store)

    # Verify chunk store record order matches the chunks file
    for i, (c, r) in enumerate(zip(chunks, chunk_store.records)):
        if c.get("members") != r.get("members"):
            print(
                f"ERROR: chunk store record {i} does not match {args.chunks}; "
                "regenerate the store from this chunks file",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build queries
    all_queries, num_excluded = build_eval_queries(chunks, dictionary)
    group_a, group_b = split_queries_by_group(all_queries, dictionary)

    print(f"Total multi-member cluster entries: {len(all_queries)}")
    print(f"Group A (has translations): {len(group_a)}")
    print(f"Group B (no translations): {len(group_b)}")
    print(f"Excluded entries (clusters without translations): {num_excluded}")
    print()

    # Get embedder from chunk-store metadata
    backend, model = resolve_embedder_config(chunk_store.meta)
    embedder = get_embedder(backend, model=model, device=args.device)

    # Build baseline word-to-indices map
    word_to_baseline_indices = build_baseline_word_to_indices(baseline_store)

    # Evaluate each group
    print("Evaluating Group A (has translations)...")
    chunk_a, baseline_a = evaluate_rank(
        group_a,
        embedder,
        chunk_store,
        baseline_store,
        word_to_baseline_indices,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("Evaluating Group B (no translations)...")
    chunk_b, baseline_b = evaluate_rank(
        group_b,
        embedder,
        chunk_store,
        baseline_store,
        word_to_baseline_indices,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Compile results
    backend_name, model_name = resolve_embedder_config(chunk_store.meta)
    results = {
        "model": f"{backend_name}:{model_name or '(default)'}",
        "num_queries_a": len(group_a),
        "num_queries_b": len(group_b),
        "num_excluded_entries": num_excluded,
        "chunk_bucket_a": chunk_a,
        "chunk_bucket_b": chunk_b,
        "baseline_bucket_a": baseline_a,
        "baseline_bucket_b": baseline_b,
    }

    # Print report
    print()
    print(format_ref_report(results))

    # Save JSON if requested
    if args.json:
        json_results = {
            "num_queries_a": results["num_queries_a"],
            "num_queries_b": results["num_queries_b"],
            "num_excluded_entries": results["num_excluded_entries"],
            "chunk_bucket_a": chunk_a.as_dict(),
            "chunk_bucket_b": chunk_b.as_dict(),
            "baseline_bucket_a": baseline_a.as_dict(),
            "baseline_bucket_b": baseline_b.as_dict(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
