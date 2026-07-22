"""CLI for generating embeddings from dictionary."""

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import Model2VecEmbedder, get_embedder
from .store import EmbeddingStore
from .passages import has_translations, make_passage
from .build_chunks import read_chunks


def _generate_from_chunks(args):
    """Embed a ref-clustered embedding_chunks.jsonl (from build_chunks).

    Each chunk's ``text`` is embedded as one passage (with ``--passage-prefix``);
    the chunk dicts themselves become the store records. ``strategy`` is
    ``"chunks"`` in the meta so downstream tools can tell the sets apart.
    """
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"ERROR: chunks not found at {chunks_path}", file=sys.stderr)
        sys.exit(1)

    chunks = read_chunks(chunks_path)
    print(f"\nLoading chunks: {chunks_path}")
    print(f"  {len(chunks)} chunks")

    embedder = get_embedder(backend=args.backend, model=args.model, device=args.device)
    print(f"Dim:          {embedder.dim}")

    texts = [args.passage_prefix + c["text"] for c in chunks]
    if texts:
        print(f"  Example: {texts[0]!r}")

    print("\nGenerating embeddings...")
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
        records=chunks,
        batch_size=args.batch_size,
        progress=progress_callback,
    )

    elapsed = time.time() - start
    print()
    print(f"  Done: {elapsed:.1f}s ({len(texts) / max(elapsed, 1e-9):.0f} chunks/s)")
    print(f"  Shape: {store.embeddings.shape}")

    try:
        bv = importlib.metadata.version(args.backend)
    except importlib.metadata.PackageNotFoundError:
        bv = ""

    store.meta = {
        "backend": args.backend,
        "model": args.model or "default",
        "provider": "local" if args.backend in ("fastembed", "model2vec", "sentence-transformers") else "api",
        "strategy": "chunks",
        "num_entries": len(chunks),
        "embedding_dim": int(store.embeddings.shape[1]),
        "passage_prefix": args.passage_prefix,
        "query_prefix": args.query_prefix,
        "backend_version": bv,
    }

    print(f"\nSaving to: {args.out}")
    store.save(args.out)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(chunks)} chunk embeddings ({store.embeddings.shape[1]}d)")
    print("=" * 60)


def main():
    """Main entry point for embedding generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from Prussian dictionary"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="../corpus/parsed/twanksta_entries.json",
        help="Path to dictionary JSON file",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help=(
            "Path to a ref-clustered embedding_chunks.jsonl (from build_chunks). "
            "When set, embeds each chunk's 'text' instead of dictionary passages."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fastembed",
        choices=["fastembed", "model2vec", "sentence-transformers", "api"],
        help="Embedding backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier. For model2vec: base HF model to distill from.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/embeddings_fastembed",
        help="Output file stem (for model2vec, also used as model directory)",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="",
        help="Prefix for embedding passages",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default="",
        help="Prefix for embedding queries (saved in store meta, applied by eval)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=256,
        help="PCA dimensions for model2vec distillation (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model2vec distillation / sentence-transformers (default: auto-detect)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow custom model code when distilling (model2vec, e.g. nomic-ai models)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)
    print(f"Backend:      {args.backend}")
    print(f"Model:        {args.model or '(default)'}")
    print(f"Strategy:     translations_only (with Prussian headword)")
    print(f"Batch size:   {args.batch_size}")
    if args.backend == "model2vec":
        print(f"PCA dims:     {args.pca_dims}")
        print(f"Device:       {args.device}")
    print("=" * 60)

    # Chunks mode: embed pre-built ref-clustered chunks instead of passages.
    if args.chunks:
        _generate_from_chunks(args)
        return

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
    print(f"  {original_count} \u2192 {len(entries)} entries (filtered by translations)")

    # model2vec backend: distill from base HF model, store artifacts alongside
    if args.backend == "model2vec":
        model_name = args.model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_dir = Path(args.out)
        device = args.device or "cpu"

        try:
            from model2vec.distill import distill
        except ImportError as exc:
            print(
                "ERROR: model2vec[distill] is required for the model2vec backend.\n"
                "Install with: pip install 'prussian-embeddings[distill]'",
                file=sys.stderr,
            )
            sys.exit(1)

        from .vocabulary import build_vocabulary, save_vocabulary

        vocabulary = build_vocabulary(entries)
        print(f"Vocabulary:   {len(vocabulary)} unique tokens")

        model_dir.mkdir(parents=True, exist_ok=True)
        save_vocabulary(vocabulary, model_dir / "vocabulary.txt")

        print(f"\nDistilling:   {model_name}")
        print(f"  vocabulary: {len(vocabulary)} words")
        print(f"  pca_dims:   {args.pca_dims}")
        print(f"  device:     {device}")
        start = time.time()
        model = distill(
            model_name=model_name,
            vocabulary=vocabulary,
            device=device,
            pca_dims=args.pca_dims,
            trust_remote_code=args.trust_remote_code,
        )
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

        model.save_pretrained(model_dir)

        distill_meta = {
            "base_model": model_name,
            "vocabulary_size": len(vocabulary),
            "pca_dims": args.pca_dims,
            "distilled_at": datetime.now(timezone.utc).isoformat(),
        }
        (model_dir / "distill_meta.json").write_text(
            json.dumps(distill_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Saved model to {model_dir}")

        embedder = Model2VecEmbedder(str(model_dir))
        model_ref = str(model_dir)
    else:
        # fastembed / sentence-transformers / api: load embedder directly
        embedder = get_embedder(backend=args.backend, model=args.model, device=args.device)
        model_ref = args.model or "default"

    embedding_dim = embedder.dim
    print(f"Dim:          {embedding_dim}")

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

    # Resolve backend version
    try:
        bv = importlib.metadata.version(args.backend)
    except importlib.metadata.PackageNotFoundError:
        bv = ""

    # Build metadata
    metadata = {
        "backend": args.backend,
        "model": model_ref,
        "provider": "local" if args.backend in ("fastembed", "model2vec", "sentence-transformers") else "api",
        "strategy": "translations_only",
        "num_entries": len(filtered_entries),
        "embedding_dim": int(store.embeddings.shape[1]),
        "passage_prefix": args.passage_prefix,
        "query_prefix": args.query_prefix,
        "backend_version": bv,
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
