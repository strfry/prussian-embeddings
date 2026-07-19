"""Evaluation of embedding backends against the dictionary."""

import argparse
import importlib.metadata
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .passages import has_translations, make_passage
from .store import EmbeddingStore

PASSAGE_LANGS = ["engl", "miks", "leit", "latt"]


@dataclass
class EvalQuery:
    text: str
    lang: str
    gold_idx: int
    seen: bool


@dataclass
class Bucket:
    n: int = 0
    hits_1: int = 0
    hits_5: int = 0
    hits_10: int = 0
    mrr_sum: float = 0.0
    any_1: int = 0

    def as_dict(self) -> dict:
        n = self.n
        if n == 0:
            return {"n": 0, "Hit@1": 0.0, "Hit@5": 0.0, "Hit@10": 0.0, "MRR": 0.0, "any@1": 0.0}
        return {
            "n": n,
            "Hit@1": self.hits_1 / n,
            "Hit@5": self.hits_5 / n,
            "Hit@10": self.hits_10 / n,
            "MRR": self.mrr_sum / n,
            "any@1": self.any_1 / n,
        }


def parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a spec string like 'model2vec:models/x' into (backend, model)."""
    if ":" in spec:
        backend, model = spec.split(":", 1)
        return backend, model
    return spec, None


def resolve_embedder_config(meta: dict) -> Tuple[str, Optional[str]]:
    """Resolve backend and model name from store metadata.

    Pure function — no side effects, easy to test.

    Args:
        meta: metadata dict (from {stem}.meta.json)

    Returns:
        (backend, model) tuple; model is None for default models.

    Raises:
        ValueError: If meta["backend"] is missing.
    """
    backend = meta.get("backend")
    if not backend:
        raise ValueError("store metadata missing 'backend' field")
    model = meta.get("model")
    if model in ("default", "", None):
        model = None
    return backend, model


def _installed_version(backend: str) -> Optional[str]:
    """Return the installed version of a backend package, or None."""
    pkg_map = {"fastembed": "fastembed", "model2vec": "model2vec", "sentence-transformers": "sentence-transformers"}
    pkg = pkg_map.get(backend)
    if not pkg:
        return None
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_entries(path: str) -> list[dict]:
    """Load dictionary entries, filtering by has_translations."""
    dict_path = Path(path)
    if not dict_path.exists():
        print(f"ERROR: Dictionary not found at {dict_path}", file=sys.stderr)
        sys.exit(1)

    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        print("ERROR: Expected list or dict in dictionary JSON", file=sys.stderr)
        sys.exit(1)

    return [e for e in entries if has_translations(e)]


def build_passages(entries: list[dict], prefix: str = "") -> Tuple[List[str], list[dict]]:
    """Build passages and records, mirroring generate.py logic."""
    texts, records = [], []
    for entry in entries:
        passage = make_passage(
            entry,
            include_prussian=True,
            langs=PASSAGE_LANGS,
            prefix=prefix,
        )
        if passage:
            texts.append(passage)
            records.append(entry)
    return texts, records


def build_eval_queries(
    records: list[dict],
    langs: List[str],
    *,
    limit: int = 0,
    seed: int = 42,
) -> List[EvalQuery]:
    """Build evaluation queries from records.

    seen = first translation of a PASSAGE_LANG (the one that appears in the passage)
    unseen = all other translations (pols, mask, second+ translations of passage langs)
    """
    rng = random.Random(seed)
    indices = list(range(len(records)))
    if limit > 0 and limit < len(indices):
        indices = sorted(rng.sample(indices, limit))

    queries: List[EvalQuery] = []
    seen_texts: Set[Tuple[str, str]] = set()

    for idx in indices:
        rec = records[idx]
        translations = rec.get("translations", {})

        for lang in langs:
            lang_translations = translations.get(lang, [])
            for t_i, text in enumerate(lang_translations):
                if not text or not text.strip():
                    continue
                text = text.strip()
                key = (lang, text.lower())
                if key in seen_texts:
                    continue
                seen_texts.add(key)

                is_seen = lang in PASSAGE_LANGS and t_i == 0
                queries.append(EvalQuery(text=text, lang=lang, gold_idx=idx, seen=is_seen))

    return queries


def build_gold_sets(records: list[dict], langs: List[str]) -> Dict[Tuple[str, str], Set[int]]:
    """Build gold sets: (lang, text.lower()) -> set of record indices.

    An entry is a gold match if any of its translations in `lang` matches (case-insensitive).
    """
    gold: Dict[Tuple[str, str], Set[int]] = {}
    for idx, rec in enumerate(records):
        translations = rec.get("translations", {})
        for lang in langs:
            for text in translations.get(lang, []):
                if not text or not text.strip():
                    continue
                key = (lang, text.strip().lower())
                gold.setdefault(key, set()).add(idx)
    return gold


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    return matrix / norms


def evaluate_spec(
    embedder,
    store,
    queries: List[EvalQuery],
    gold_sets: Dict[Tuple[str, str], Set[int]],
    *,
    ks: List[int] = (1, 5, 10),
    batch_size: int = 256,
    query_prefix: str = "",
) -> dict:
    """Evaluate a spec against queries.

    Returns dict with keys: 'build_time', 'query_time', 'num_queries',
    and 'results' -> dict of (lang, subset) -> Bucket.
    """
    if not queries:
        return {"build_time": 0, "query_time": 0, "num_queries": 0, "results": {}}

    # Batch-embed all queries
    texts = [query_prefix + q.text if query_prefix else q.text for q in queries]
    start = time.time()
    q_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        q_vecs.append(embedder.get_embeddings(batch))
    q_vecs = np.vstack(q_vecs)
    query_time = time.time() - start

    # L2-normalize query vectors
    q_vecs = _l2_normalize(q_vecs)

    # Scores: (n_queries, n_passages)
    scores = q_vecs @ store.embeddings.T
    scores = np.clip(scores, -1.0, 1.0)

    max_k = max(ks)

    # Buckets keyed by (lang, subset)
    buckets: Dict[Tuple[str, str], Bucket] = {}

    for qi, q in enumerate(queries):
        row = scores[qi]
        gold = q.gold_idx
        gold_score = row[gold]

        # Strict rank: how many passages score strictly higher
        rank_strict = int(np.sum(row > gold_score)) + 1

        # Any rank: best rank among all gold-set passages
        key = (q.lang, q.text.lower())
        gset = gold_sets.get(key, set())
        if gset:
            g_indices = np.array(sorted(gset))
            g_scores = row[g_indices]
            any_rank = int(np.min(np.sum(row[:, None] > g_scores[None, :], axis=0))) + 1
        else:
            any_rank = rank_strict

        subset = "seen" if q.seen else "unseen"
        bucket_key = (q.lang, subset)
        if bucket_key not in buckets:
            buckets[bucket_key] = Bucket()
        b = buckets[bucket_key]
        b.n += 1
        if rank_strict <= 1:
            b.hits_1 += 1
        if rank_strict <= 5:
            b.hits_5 += 1
        if rank_strict <= 10:
            b.hits_10 += 1
        b.mrr_sum += 1.0 / rank_strict
        if any_rank <= 1:
            b.any_1 += 1

    # Build per-language "all" buckets
    all_langs = sorted({k[0] for k in buckets})
    for lang in all_langs:
        seen_b = buckets.get((lang, "seen"), Bucket())
        unseen_b = buckets.get((lang, "unseen"), Bucket())
        all_b = Bucket(
            n=seen_b.n + unseen_b.n,
            hits_1=seen_b.hits_1 + unseen_b.hits_1,
            hits_5=seen_b.hits_5 + unseen_b.hits_5,
            hits_10=seen_b.hits_10 + unseen_b.hits_10,
            mrr_sum=seen_b.mrr_sum + unseen_b.mrr_sum,
            any_1=seen_b.any_1 + unseen_b.any_1,
        )
        buckets[(lang, "all")] = all_b

    # Grand total
    total = Bucket()
    for (lang, subset), b in list(buckets.items()):
        if subset == "all":
            total.n += b.n
            total.hits_1 += b.hits_1
            total.hits_5 += b.hits_5
            total.hits_10 += b.hits_10
            total.mrr_sum += b.mrr_sum
            total.any_1 += b.any_1
    buckets[("ALL", "all")] = total

    return {
        "build_time": 0,
        "query_time": query_time,
        "num_queries": len(queries),
        "results": buckets,
    }


def format_report(spec: str, meta: dict, results: dict) -> str:
    """Format evaluation results as a fixed-width table."""
    backend, model = parse_spec(spec)
    dim = meta.get("embedding_dim", "?")
    model_name = meta.get("model", model or "(default)")

    lines = []
    lines.append(f"=== {backend} ({model_name}, {dim}d) ===")

    query_time = results.get("query_time", 0)
    n_queries = results.get("num_queries", 0)
    if query_time > 0 and n_queries > 0:
        lines.append(f"queries: {n_queries} in {query_time:.1f}s ({n_queries / query_time:.0f} q/s)")

    header = f"{'lang':<6} {'subset':<8} {'n':>6} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7} {'any@1':>7}"
    lines.append(header)

    buckets = results.get("results", {})
    seen_order = ["seen", "unseen", "all"]
    all_langs = sorted({k[0] for k in buckets if k[0] != "ALL"})

    for lang in all_langs:
        for subset in seen_order:
            key = (lang, subset)
            if key not in buckets:
                continue
            d = buckets[key].as_dict()
            lines.append(
                f"{lang:<6} {subset:<8} {d['n']:>6} {d['Hit@1']:>7.3f} {d['Hit@5']:>7.3f} "
                f"{d['Hit@10']:>7.3f} {d['MRR']:>7.3f} {d['any@1']:>7.3f}"
            )

    if ("ALL", "all") in buckets:
        d = buckets[("ALL", "all")].as_dict()
        lines.append(
            f"{'ALL':<6} {'all':<8} {d['n']:>6} {d['Hit@1']:>7.3f} {d['Hit@5']:>7.3f} "
            f"{d['Hit@10']:>7.3f} {d['MRR']:>7.3f} {d['any@1']:>7.3f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding backends against dictionary"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="../../corpus/parsed/twanksta_entries.json",
        help="Path to dictionary JSON file (only used in --spec mode)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        action="append",
        default=None,
        help="Backend spec (backend[:model]), repeatable. Used for on-the-fly index build.",
    )
    parser.add_argument(
        "--store",
        type=str,
        action="append",
        default=None,
        help="Path stem of a pre-generated embedding store (repeatable). "
        "Embedder is resolved from meta.json. Primary mode.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="engl,miks,leit,latt,pols,mask",
        help="Comma-separated query languages",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of entries sampled (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default=None,
        help="Prefix for query texts. In --store mode, defaults to the store's "
        "saved query_prefix (override with an explicit value; pass '' to force none).",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="",
        help="Prefix for passage texts (only used when building index in --spec mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for torch-based backends (auto-detects if None)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to save full result dict as JSON",
    )

    args = parser.parse_args()

    stores = args.store or []
    specs = args.spec or []
    langs = [l.strip() for l in args.langs.split(",")]

    # --- mode validation ---
    if stores and specs:
        print("ERROR: --store and --spec cannot be used together", file=sys.stderr)
        sys.exit(1)
    if not stores and not specs:
        print(
            "ERROR: specify at least one --store (pre-generated) or --spec (on-the-fly). "
            "See --help.",
            file=sys.stderr,
        )
        sys.exit(1)

    from .backends import get_embedder

    store_mode = bool(stores)
    all_results = {}

    if store_mode:
        # ---- Store mode: load pre-generated artefacts ----
        for stem in stores:
            print(f"\nLoading store: {stem}")
            store = EmbeddingStore.load(stem)
            backend, model = resolve_embedder_config(store.meta)
            print(f"  backend={backend}  model={model or '(default)'}")

            # backend_version consistency warning
            meta_bv = store.meta.get("backend_version")
            inst_v = _installed_version(backend)
            if meta_bv and inst_v and meta_bv != inst_v:
                print(
                    f"  WARNING: installed {backend}=={inst_v} != "
                    f"meta backend_version={meta_bv}; "
                    f"artefacts may be incompatible",
                    file=sys.stderr,
                )

            embedder = get_embedder(backend=backend, model=model, device=args.device)

            # Dim sanity check
            emb_dim = int(store.embeddings.shape[1])
            if hasattr(embedder, "dim") and embedder.dim != emb_dim:
                print(
                    f"ERROR: embedder dim={embedder.dim} != store dim={emb_dim}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  {store.embeddings.shape[0]} entries, dim={emb_dim}")

            records = store.records
            queries = build_eval_queries(records, langs, limit=args.limit, seed=args.seed)
            gold_sets = build_gold_sets(records, langs)

            qp = (
                args.query_prefix
                if args.query_prefix is not None
                else store.meta.get("query_prefix", "")
            )
            print(f"  query_prefix={qp!r}")

            result = evaluate_spec(
                embedder,
                store,
                queries,
                gold_sets,
                batch_size=args.batch_size,
                query_prefix=qp,
            )

            label = f"{backend}:{model or 'default'} [{Path(stem).name}]"
            meta = store.meta

            report = format_report(label, meta, result)
            print(f"\n{report}")

            all_results[label] = {
                "meta": meta,
                "num_queries": result["num_queries"],
                "query_time": result["query_time"],
                "results": {
                    f"{k[0]},{k[1]}": v.as_dict()
                    for k, v in result["results"].items()
                },
            }

    else:
        # ---- Spec mode: on-the-fly index build ----
        print(f"Loading dictionary: {args.dictionary}")
        entries = load_entries(args.dictionary)
        print(f"  {len(entries)} entries")

        for spec in specs:
            backend, model = parse_spec(spec)
            embedder = get_embedder(backend=backend, model=model, device=args.device)

            print(f"\nBuilding index for {spec}...")
            start = time.time()
            texts, records = build_passages(entries, prefix=args.passage_prefix)
            store = EmbeddingStore.build(
                embedder,
                texts=texts,
                records=records,
                batch_size=args.batch_size,
            )
            build_time = time.time() - start
            dim = int(store.embeddings.shape[1])
            qp = args.query_prefix or ""
            store.meta = {
                "backend": backend,
                "model": model or "default",
                "embedding_dim": dim,
                "passage_prefix": args.passage_prefix,
                "query_prefix": qp,
            }
            print(f"  build: {build_time:.1f}s ({len(texts) / build_time:.0f} p/s)  dim={dim}")

            records = store.records
            queries = build_eval_queries(records, langs, limit=args.limit, seed=args.seed)
            gold_sets = build_gold_sets(records, langs)

            result = evaluate_spec(
                embedder,
                store,
                queries,
                gold_sets,
                batch_size=args.batch_size,
                query_prefix=qp,
            )
            result["build_time"] = build_time
            meta = store.meta

            report = format_report(spec, meta, result)
            print(f"\n{report}")

            all_results[spec] = {
                "meta": meta,
                "num_queries": result["num_queries"],
                "query_time": result["query_time"],
                "results": {
                    f"{k[0]},{k[1]}": v.as_dict()
                    for k, v in result["results"].items()
                },
            }

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {args.json}")


if __name__ == "__main__":
    main()
