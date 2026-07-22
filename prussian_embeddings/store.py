"""Embedding storage and query operations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EmbeddingStore:
    """Generic embedding store with save/load/query operations.
    
    File format is byte-compatible with existing artifacts:
    - {stem}.embeddings.npy: numpy array (n, dim) float32
    - {stem}.entries.json: list of record dicts
    - {stem}.meta.json: metadata dict (optional)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        records: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Initialize store.
        
        Args:
            embeddings: Array of shape (n, dim), dtype float32
            records: List of record dicts (one per embedding)
            meta: Optional metadata dict
        """
        self.embeddings = embeddings
        self.records = records
        self.meta = meta or {}

    @classmethod
    def build(
        cls,
        embedder: Any,
        texts: List[str],
        records: List[Dict[str, Any]],
        *,
        batch_size: int = 256,
        meta: Optional[Dict[str, Any]] = None,
        progress: Optional[Any] = None,
    ) -> "EmbeddingStore":
        """Build store by embedding texts.
        
        Args:
            embedder: Embedder instance with get_embeddings() method
            texts: List of texts to embed
            records: List of records (one per text)
            batch_size: Batch size for embedding
            meta: Optional metadata dict
            progress: Optional progress callback
            
        Returns:
            EmbeddingStore instance
        """
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = embedder.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            if progress:
                batch_num = (i // batch_size) + 1
                progress(batch_num, num_batches, len(all_embeddings))

        embeddings = np.array(all_embeddings, dtype=np.float32)
        return cls(embeddings=embeddings, records=records, meta=meta or {})

    @classmethod
    def load(cls, stem: str) -> "EmbeddingStore":
        """Load store from files.
        
        Args:
            stem: Path stem (files will be {stem}.embeddings.npy, etc.)
            
        Returns:
            EmbeddingStore instance
            
        Raises:
            FileNotFoundError: If .npy or .json files don't exist
        """
        stem = str(stem)
        
        emb_file = Path(f"{stem}.embeddings.npy")
        entries_file = Path(f"{stem}.entries.json")
        meta_file = Path(f"{stem}.meta.json")
        
        if not emb_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_file}")
        if not entries_file.exists():
            raise FileNotFoundError(f"Entries not found: {entries_file}")
        
        embeddings = np.load(emb_file)
        
        with open(entries_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        
        meta = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        return cls(embeddings=embeddings, records=records, meta=meta)

    def save(self, stem: str) -> None:
        """Save store to files.
        
        Args:
            stem: Path stem (files will be {stem}.embeddings.npy, etc.)
        """
        stem = str(stem)
        stem_path = Path(stem)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(f"{stem}.embeddings.npy", self.embeddings)
        
        with open(f"{stem}.entries.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        
        with open(f"{stem}.meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def top_k(
        self, query_vec: np.ndarray, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar embeddings via cosine similarity.
        
        Args:
            query_vec: Query embedding, shape (dim,), float32
            k: Number of results to return
            
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        # Cosine similarity: (embeddings @ query) / (||embeddings|| * ||query||)
        # Since embeddings are L2-normalized, ||embeddings|| = 1
        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            query_norm = 1.0
        query_normalized = query_vec / query_norm
        
        # Cosine similarity with normalized embeddings
        scores = self.embeddings @ query_normalized
        
        # Clip to [-1, 1] to be safe before arccos
        scores = np.clip(scores, -1.0, 1.0)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def query(
        self,
        embedder: Any,
        text: str,
        k: int = 10,
        *,
        query_prefix: str = "",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Query store with text.
        
        Args:
            embedder: Embedder instance with get_embedding() method
            text: Query text
            k: Number of results to return
            query_prefix: Optional prefix to prepend to query text
            
        Returns:
            List of (record, score) tuples, sorted by score descending
        """
        query_text = query_prefix + text if query_prefix else text
        query_vec = embedder.get_embedding(query_text, is_query=True)
        
        top_results = self.top_k(query_vec, k=k)
        
        results = []
        for idx, score in top_results:
            if idx < len(self.records):
                results.append((self.records[idx], score))
        
        return results

    @classmethod
    def load_with_embedder(
        cls,
        stem: str,
        *,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Tuple["EmbeddingStore", Any]:
        """Load store and create matching embedder from metadata.

        Returns (store, embedder).
        Raises ValueError on dim mismatch.
        """
        from .backends import get_embedder

        store = cls.load(stem)
        backend, model = resolve_embedder_config(store.meta)
        embedder = get_embedder(
            backend=backend, model=model,
            device=device, trust_remote_code=trust_remote_code,
        )
        emb_dim = int(store.embeddings.shape[1])
        if hasattr(embedder, "dim") and embedder.dim != emb_dim:
            raise ValueError(
                f"embedder dim={embedder.dim} != store dim={emb_dim}"
            )
        return store, embedder


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
