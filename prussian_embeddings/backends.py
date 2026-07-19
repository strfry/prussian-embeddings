"""Embedding backends: fastembed, model2vec, and API."""

import sys
from typing import List, Optional, Protocol

import numpy as np

from .config import env_config


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe for zero vectors)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    return matrix / norms


class Embedder(Protocol):
    """Protocol for embedding backends."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (len(texts), dim), dtype float32, L2-normalized
        """
        ...

    def get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text.
        
        Args:
            text: Text string
            
        Returns:
            Array of shape (dim,), dtype float32, L2-normalized
        """
        ...


class FastEmbedEmbedder:
    """Local embeddings via fastembed (ONNX/CPU)."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        """Initialize fastembed embedder.

        Args:
            model_name: Hugging Face model ID (must be in TextEmbedding.list_supported_models())
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the 'fastembed' embedding backend. "
                "Install it with `pip install prussian-embeddings[local]`."
            ) from exc

        print(f"Loading embedding model: {model_name}...", file=sys.stderr)
        self.model = TextEmbedding(model_name=model_name)
        self.dim = self.model.embedding_size

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts. Returns an (n, dim) L2-normalized float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # fastembed returns embeddings as a generator of lists
        embeddings = list(self.model.embed(texts))
        return _l2_normalize(np.array(embeddings, dtype=np.float32))

    def get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a (dim,) L2-normalized float32 array."""
        return self.get_embeddings([text])[0]


class Model2VecEmbedder:
    """Local static embeddings via model2vec (CPU-only, no network)."""

    def __init__(self, model_name: str = "minishlab/potion-multilingual-128M"):
        """Initialize model2vec embedder.
        
        Args:
            model_name: Hugging Face id or local directory path
        """
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise ImportError(
                "model2vec is required for the 'model2vec' embedding backend. "
                "Install it with `pip install prussian-embeddings[model2vec]`."
            ) from exc

        print(f"Loading embedding model: {model_name}...", file=sys.stderr)
        self.model = StaticModel.from_pretrained(model_name)
        self.dim = int(self.model.dim)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts. Returns an (n, dim) L2-normalized float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(texts)
        return _l2_normalize(embeddings)

    def get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a (dim,) L2-normalized float32 array."""
        return self.get_embeddings([text])[0]


class ApiEmbedder:
    """Remote embedding backend via EmbeddingClient."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        dim: int = 0,
    ):
        """Initialize API embedder.
        
        Args:
            api_key: API authentication key
            base_url: API endpoint base URL
            model: Model identifier on the API
            dim: Embedding dimension
        """
        from .client import EmbeddingClient

        if not api_key:
            raise ValueError(
                "api_key is required for the 'api' embedding backend. "
                "Set via API_KEY environment variable."
            )

        self.client = EmbeddingClient(
            api_key=api_key,
            base_url=base_url,
            embedding_model=model,
            embedding_dim=dim,
        )
        self.dim = dim if dim > 0 else self.client.embedding_dim

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts via API."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.client.get_embeddings(texts)

    def get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text via API."""
        return self.client.get_embedding(text)


def get_embedder(
    backend: Optional[str] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    dim: Optional[int] = None,
) -> Embedder:
    """Get an embedder for the specified (or configured) backend.
    
    Args:
        backend: "fastembed" (default), "model2vec", or "api"
        model: Model identifier (overrides env config)
        api_key: API key for 'api' backend (overrides env config)
        base_url: API base URL (overrides env config)
        dim: Embedding dimension (overrides env config)
        
    Returns:
        Embedder instance
        
    Raises:
        ValueError: If backend is unknown or required params are missing
    """
    config = env_config()
    
    # Explicit args win; gaps filled by env_config()
    backend = (backend or config.backend or "fastembed").lower()
    model = model or config.model
    api_key = api_key or config.api_key
    base_url = base_url or config.base_url
    dim = dim if dim is not None else config.dim
    
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name=model)
    elif backend == "model2vec":
        return Model2VecEmbedder(model_name=model)
    elif backend == "api":
        return ApiEmbedder(api_key=api_key, base_url=base_url, model=model, dim=dim)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND: {backend!r} "
            "(expected 'fastembed', 'model2vec', or 'api')"
        )
