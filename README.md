# prussian-embeddings

Embeddings functionality for the Prussian dictionary: semantic search backends, embedding store, and generation CLI.

## Features

- **Multiple backends**: fastembed (ONNX/CPU, default), model2vec (local), API (remote)
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

# Development
pip install prussian-embeddings[dev]
```

## Supported Backends

### fastembed (default)
Local ONNX-based embeddings, CPU-only, no network required.
- Model: `intfloat/multilingual-e5-small` (default)
- Dimension: 384

### model2vec
Lightweight static embeddings.
- Model: `minishlab/potion-multilingual-128M` (default)
- Dimension: 128

### API
Remote OpenAI-compatible or Jina-compatible embedding service.
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

```bash
prussian-embeddings-generate \
    --dictionary dict.json \
    --backend fastembed \
    --model intfloat/multilingual-e5-small \
    --out embeddings
```

## Environment Variables

- `EMBEDDING_BACKEND` – "fastembed" (default), "model2vec", or "api"
- `EMBEDDING_MODEL` – model identifier
- `EMBEDDING_DIM` – output dimension (inferred from model if not set)
- `RERANKER_MODEL` – reranker model (for API backend)
- `API_KEY` / `JINA_API_KEY` – API authentication
- `API_BASE_URL` – API endpoint
- `QUERY_PREFIX` – query text prefix (for asymmetric search)
- `PASSAGE_PREFIX` – passage text prefix

## License

MIT (or consult LICENSE in the repository)
