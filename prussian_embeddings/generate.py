"""CLI for generating embeddings from dictionary."""

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import get_embedder
from .store import EmbeddingStore
from .passages import has_translations, make_passage


def main():
    """Main entry point for embedding generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from Prussian dictionary"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="../../corpus/parsed/twanksta_entries.json",
        help="Path to dictionary JSON file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fastembed",
        choices=["fastembed", "model2vec", "api"],
        help="Embedding backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (overrides env/default)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/embeddings_fastembed",
        help="Output file stem",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="Document: ",
        help="Prefix for embedding passages",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)
    print(f"Backend:      {args.backend}")
    print(f"Model:        {args.model or '(default)'}")
    print(f"Strategy:     translations_only (with Prussian headword)")
    print(f"Batch size:   {args.batch_size}")
    print("=" * 60)

    # Load embedder
    embedder = get_embedder(backend=args.backend, model=args.model)
    embedding_dim = embedder.dim
    print(f"Dim:          {embedding_dim}")

    # Load dictionary
    dict_path = Path(args.dictionary)
    print(f"\nLoading dictionary: {dict_path}")
    
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

    original_count = len(entries)
    entries = [e for e in entries if has_translations(e)]
    print(f"  {original_count} → {len(entries)} entries (filtered by translations)")

    # Generate passages
    texts = []
    filtered_entries = []
    for entry in entries:
        passage = make_passage(
            entry,
            include_prussian=True,
            langs=["engl", "miks", "leit", "latt"],
            prefix=args.passage_prefix,
        )
        if passage:
            texts.append(passage)
            filtered_entries.append(entry)

    print(f"  {len(texts)} passages prepared")
    if texts:
        print(f"  Example: {texts[0]}")
        if len(texts) > 100:
            print(f"  Example: {texts[100]}")
        if len(texts) > 1000:
            print(f"  Example: {texts[1000]}")

    # Generate embeddings
    print(f"\nGenerating embeddings...")
    start = time.time()

    def progress_callback(batch_num, num_batches, total):
        pct = (batch_num / num_batches) * 100
        print(
            f"  [{pct:5.1f}%] Batch {batch_num}/{num_batches} ({total} embeddings)",
            end="\r",
        )

    store = EmbeddingStore.build(
        embedder,
        texts=texts,
        records=filtered_entries,
        batch_size=args.batch_size,
        progress=progress_callback,
    )
    
    elapsed = time.time() - start
    print()
    print(f"  Done: {elapsed:.1f}s ({len(texts) / elapsed:.0f} entries/s)")
    print(f"  Shape: {store.embeddings.shape}")

    # Build metadata
    metadata = {
        "backend": args.backend,
        "model": args.model or "default",
        "provider": "local" if args.backend in ("fastembed", "model2vec") else "api",
        "strategy": "translations_only",
        "num_entries": len(filtered_entries),
        "embedding_dim": int(store.embeddings.shape[1]),
        "passage_prefix": args.passage_prefix,
    }

    store.meta = metadata

    # Save
    output_path = args.out
    print(f"\nSaving to: {output_path}")
    store.save(output_path)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(filtered_entries)} embeddings ({store.embeddings.shape[1]}d)")
    print(f"  - {output_path}.embeddings.npy")
    print(f"  - {output_path}.entries.json")
    print(f"  - {output_path}.meta.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
