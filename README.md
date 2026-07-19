# prussian-embeddings

Embeddings functionality for the Prussian dictionary: semantic search backends, embedding store, and generation CLI.

## Features

- **Multiple backends**: fastembed (ONNX/CPU, default), sentence-transformers (torch, XPU-capable), model2vec (local), API (remote)
- **Embedding store**: Generic store with load/save/query operations
- **Dictionary support**: Passage generation for Prussian words with translations
- **CLI**: `prussian-embeddings-generate` for batch embedding generation

## Installation

```bash
# Core package (no backends)
pip install prussian-embeddings

# With fastembed (ONNX, local CPU)
pip install prussian-embeddings[local]

# With model2vec
pip install prussian-embeddings[model2vec]

# With model2vec distillation support (torch required)
pip install prussian-embeddings[distill]

# With sentence-transformers (torch, XPU-capable)
pip install prussian-embeddings[sentence-transformers]

# Development
pip install prussian-embeddings[dev]
```

## Supported Backends

### fastembed (default)
Local ONNX-based embeddings, CPU-only, no network required.
- Model: `intfloat/multilingual-e5-small` (default)
- Dimension: 384

### sentence-transformers
Torch-based embeddings with auto-XPU support (xpu → cuda → cpu), ~identical to fastembed
(same models, same pooling, L2-normalized output). Use this for fast generation of large
models like `intfloat/multilingual-e5-large` on Intel GPU.
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default)
- Dimension: 384
- Device: auto-detected, override with `--device xpu|cuda|cpu`
- Install with `pip install prussian-embeddings[sentence-transformers]`

### model2vec
Lightweight static embeddings via distillation.
- Model: distilled from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default; `--model` for different base model)
- Install with `pip install prussian-embeddings[distill]` for distillation support

### API
Remote OpenAI-compatible or Voyage-compatible embedding service.
- Requires: `API_KEY` and `API_BASE_URL` environment variables

## Usage

```python
from prussian_embeddings import get_embedder, EmbeddingStore

# Get an embedder
embedder = get_embedder(backend="fastembed")

# Generate embeddings
texts = ["Hello world", "Guten Tag"]
embeddings = embedder.get_embeddings(texts)

# Build and save a store
store = EmbeddingStore.build(
    embedder,
    texts=texts,
    records=[{"text": t} for t in texts]
)
store.save("my_embeddings")

# Load and query
store = EmbeddingStore.load("my_embeddings")
results = store.query(embedder, "hello", k=5)
```

## CLI

### Generate embeddings

```bash
prussian-embeddings-generate \
    --dictionary dict.json \
    --backend fastembed \
    --model intfloat/multilingual-e5-small \
    --out embeddings
```

### Generate embeddings (model2vec with distillation)

The `model2vec` backend distills a static model from a base HF model and produces both the model directory and the embedding store in one step:

```bash
# One command: distill + generate store
prussian-embeddings-generate \
    --dictionary ../corpus/parsed/twanksta_entries.json \
    --backend model2vec \
    --out data/embeddings_model2vec

# With a different base model
prussian-embeddings-generate \
    --dictionary ../corpus/parsed/twanksta_entries.json \
    --backend model2vec --model intfloat/multilingual-e5-small \
    --out data/embeddings_model2vec_e5

# Pass --pca-dims and --device for distillation tuning
prussian-embeddings-generate \
    --dictionary dict.json \
    --backend model2vec --pca-dims 128 --device cuda \
    --out data/embeddings_model2vec
```

### Generate embeddings (sentence-transformers, XPU)

The `sentence-transformers` backend runs the full transformer via torch and
auto-detects the best device (xpu → cuda → cpu). Vectors match fastembed's up
to float noise.

```bash
# e5-large on Intel GPU
uv run --extra sentence-transformers prussian-embeddings-generate \
    --backend sentence-transformers --model intfloat/multilingual-e5-large \
    --passage-prefix "passage: " --query-prefix "query: " \
    --device xpu --out data/embeddings_st_e5large

# Eval comparing fastembed vs sentence-transformers e5 stores
uv run --extra local --extra sentence-transformers prussian-embeddings-eval \
    --store data/embeddings_fastembed_e5large --store data/embeddings_st_e5large
```

### Evaluate

```bash
prussian-embeddings-eval \
    --store data/embeddings_fastembed \
    --store data/embeddings_m2v_minilm
```

`generate` persists both `passage_prefix` and `query_prefix` in each store's
`*.meta.json`. In `--store` mode, `eval` auto-applies the store's saved
`query_prefix` so a single comparison run can mix prefixed (e.g. full e5 with
`query: `) and unprefixed (e.g. distilled static) stores. Override with an
explicit `--query-prefix` (pass `""` to force no prefix).

`generate` persists both `--passage-prefix` and `--query-prefix` in the store meta
(`{stem}.meta.json`). In `--store` mode, eval auto-applies each store's saved
`query_prefix`; pass `--query-prefix` to override (use `--query-prefix ''` to force
none). This lets one eval run compare a full e5 model (needing `query: ` / `passage: `
prefixes) against an unprefixed distillate without a global flag.

## Environment Variables

- `EMBEDDING_BACKEND` – "fastembed" (default), "model2vec", "sentence-transformers", or "api"
- `EMBEDDING_MODEL` – model identifier
- `EMBEDDING_DIM` – output dimension (inferred from model if not set)
- `EMBEDDING_DEVICE` – device for torch-based backends (default: auto-detect)
- `RERANKER_MODEL` – reranker model (for API backend)
- `API_KEY` / `JINA_API_KEY` – API authentication
- `API_BASE_URL` – API endpoint
- `QUERY_PREFIX` – query text prefix (for asymmetric search)
- `PASSAGE_PREFIX` – passage text prefix

## License

MIT (or consult LICENSE in the repository)
