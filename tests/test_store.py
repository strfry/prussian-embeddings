"""Tests for EmbeddingStore."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from prussian_embeddings.store import EmbeddingStore


class FakeEmbedder:
    """Fake embedder for testing."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._counter = 0

    def get_embeddings(self, texts):
        """Return deterministic fake embeddings."""
        embeddings = []
        for text in texts:
            # Deterministic based on text length
            seed = len(text) + self._counter
            np.random.seed(seed)
            vec = np.random.randn(self.dim).astype(np.float32)
            # L2 normalize
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings.append(vec)
            self._counter += 1
        return np.array(embeddings, dtype=np.float32)

    def get_embedding(self, text):
        """Return single fake embedding."""
        return self.get_embeddings([text])[0]


def test_store_build_and_query():
    """Test building and querying a store."""
    embedder = FakeEmbedder(dim=8)
    
    texts = ["hello world", "good morning", "how are you"]
    records = [{"id": i, "text": t} for i, t in enumerate(texts)]
    
    store = EmbeddingStore.build(embedder, texts, records)
    
    # Check dimensions
    assert store.embeddings.shape == (3, 8)
    assert len(store.records) == 3
    
    # Check that embeddings are normalized
    norms = np.linalg.norm(store.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_store_save_and_load():
    """Test saving and loading a store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = FakeEmbedder(dim=8)
        
        texts = ["hello", "world"]
        records = [{"id": 0}, {"id": 1}]
        meta = {"version": "1.0"}
        
        store = EmbeddingStore.build(embedder, texts, records, meta=meta)
        
        stem = str(Path(tmpdir) / "test")
        store.save(stem)
        
        # Verify files exist
        assert Path(f"{stem}.embeddings.npy").exists()
        assert Path(f"{stem}.entries.json").exists()
        assert Path(f"{stem}.meta.json").exists()
        
        # Load and verify
        loaded = EmbeddingStore.load(stem)
        
        np.testing.assert_array_equal(loaded.embeddings, store.embeddings)
        assert loaded.records == store.records
        assert loaded.meta == meta


def test_store_top_k():
    """Test top-k similarity search."""
    embedder = FakeEmbedder(dim=8)
    
    texts = ["a", "b", "c", "d"]
    records = [{"id": i} for i in range(4)]
    
    store = EmbeddingStore.build(embedder, texts, records)
    
    # Query with first embedding
    query_vec = store.embeddings[0]
    top_k = store.top_k(query_vec, k=2)
    
    assert len(top_k) == 2
    assert top_k[0][0] == 0  # First result should be most similar (itself)
    assert top_k[0][1] >= top_k[1][1]  # Scores should be descending


def test_store_query():
    """Test query method."""
    embedder = FakeEmbedder(dim=8)
    
    texts = ["test1", "test2", "test3"]
    records = [{"id": i, "text": t} for i, t in enumerate(texts)]
    
    store = EmbeddingStore.build(embedder, texts, records)
    
    results = store.query(embedder, "query", k=2)
    
    assert len(results) <= 2
    for record, score in results:
        assert "id" in record
        assert isinstance(score, float)


def test_store_query_with_prefix():
    """Test query with prefix."""
    embedder = FakeEmbedder(dim=8)
    
    texts = ["test"]
    records = [{"id": 0}]
    
    store = EmbeddingStore.build(embedder, texts, records)
    
    results = store.query(embedder, "query", k=1, query_prefix="prefix: ")
    
    assert len(results) == 1


def test_store_empty():
    """Test empty store."""
    embeddings = np.zeros((0, 8), dtype=np.float32)
    records = []
    
    store = EmbeddingStore(embeddings, records)
    
    assert store.embeddings.shape == (0, 8)
    assert len(store.records) == 0
    
    query_vec = np.ones(8, dtype=np.float32)
    top_k = store.top_k(query_vec, k=5)
    assert len(top_k) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
