"""CLI: prussian-embeddings-search — hybrid dense+BM25 search with optional reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_STORE_STEM = "data/embeddings_st_e5large"


def _record_text(rec: dict) -> str:
    """Extract display/BM25 text from a record (chunk or entry)."""
    if "text" in rec:
        return rec["text"]
    from prussian_embeddings.passages import make_passage

    return make_passage(rec, include_prussian=True)


def _build_bm25(records):
    from prussian_embeddings.hybrid import BM25Index

    texts = [_record_text(r) for r in records]
    return BM25Index(texts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prussian-embeddings-search",
        description="Hybrid dense + BM25 search in the Prussian dictionary.",
    )
    ap.add_argument("query", help="Search query (German, English, etc.).")
    ap.add_argument(
        "--store",
        default=DEFAULT_STORE_STEM,
        help=f"EmbeddingStore stem path (default: {DEFAULT_STORE_STEM})",
    )
    ap.add_argument("--k", type=int, default=10, help="Number of results (default: 10).")
    ap.add_argument(
        "--query-prefix",
        default="query: ",
        help='Prefix for the dense query (default: "query: ").',
    )
    ap.add_argument(
        "--dense-only",
        action="store_true",
        help="Use only dense retrieval (skip BM25).",
    )
    ap.add_argument(
        "--bm25-only",
        action="store_true",
        help="Use only BM25 retrieval (skip dense).",
    )
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (requires fastembed).",
    )
    ap.add_argument(
        "--context",
        default=None,
        help="Reranking context (enables --rerank implicitly when set).",
    )
    ap.add_argument("--json", action="store_true", help="Emit raw JSON output.")
    ap.add_argument("--verbose", action="store_true", help="Print tracebacks on errors.")
    args = ap.parse_args(argv)

    if args.context:
        args.rerank = True

    # ── Load store + embedder ──────────────────────────────────────────────
    from prussian_embeddings.store import EmbeddingStore, resolve_embedder_config

    try:
        store, embedder = EmbeddingStore.load_with_embedder(
            args.store, trust_remote_code=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    backend, model = resolve_embedder_config(store.meta)
    records = store.records
    emb_dim = int(store.embeddings.shape[1])

    print(f"store:   {args.store}")
    print(f"  backend={backend}  model={model or '(default)'}")
    print(f"  {len(records)} records, dim={emb_dim}")

    # ── BM25 index ────────────────────────────────────────────────────────
    bm25 = _build_bm25(records) if not args.dense_only else None

    # ── Query prefix from meta ────────────────────────────────────────────
    qp = args.query_prefix if args.query_prefix is not None else store.meta.get("query_prefix", "")
    print(f"  query_prefix={qp!r}")

    if args.bm25_only:
        bm25_hits = bm25.query(args.query, k=args.k)
        results = [(records[idx], score) for idx, score in bm25_hits]
    elif args.dense_only:
        results = store.query(embedder, args.query, k=args.k, query_prefix=qp)
    else:
        from prussian_embeddings.hybrid import hybrid_query

        results = hybrid_query(
            store,
            embedder,
            bm25,
            args.query,
            k=args.k,
            query_prefix=qp,
        )

    # ── Reranking ─────────────────────────────────────────────────────────
    best_line_map: dict[int, str] = {}
    if args.rerank:
        from prussian_embeddings import annotate_chunk, get_reranker

        reranker = get_reranker()
        context = args.context or args.query
        reranked = []
        for rec, score in results:
            annotated = annotate_chunk(rec, context, reranker)
            reranked.append((annotated, score))
            best_line_map[id(annotated)] = annotated.get("best_line", "")
        results = reranked

    # ── Output ────────────────────────────────────────────────────────────
    if args.json:
        import json

        out = []
        for rec, score in results:
            entry = dict(rec) if isinstance(rec, dict) else {"text": str(rec)}
            entry["_text"] = _record_text(rec) if isinstance(rec, dict) else str(rec)
            entry["score"] = score
            out.append(entry)
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    else:
        if not results:
            print("no results.")
            return 0

        for i, (rec, score) in enumerate(results, 1):
            text = _record_text(rec) if isinstance(rec, dict) else str(rec)
            first_line = text.split("\n", 1)[0]
            print(f"  {i:2d}. [{score:.4f}] {first_line}")

            if args.rerank:
                bl = best_line_map.get(id(rec), "")
                if bl and bl != first_line:
                    print(f"      → {bl}")

            if not args.rerank and "\n" in text:
                for subline in text.split("\n")[1:3]:
                    print(f"          {subline}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
