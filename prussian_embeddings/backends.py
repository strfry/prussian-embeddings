"""Embedding backends: fastembed, model2vec, sentence-transformers, and API."""

import sys
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .config import env_config


def _pick_device(device: Optional[str] = None) -> str:
    """Pick the best available device: explicit > XPU > CUDA > CPU."""
    if device and device != "auto":
        return device
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


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

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
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

    def __init__(self, model_name: str):
        """Initialize model2vec embedder.

        Args:
            model_name: Hugging Face id or local directory path (required)
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


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (torch, XPU-capable)."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the 'sentence-transformers' embedding backend. "
                "Install it with `pip install prussian-embeddings[sentence-transformers]`."
            ) from exc

        self.device = _pick_device(device)
        print(f"Loading model: {model_name} on {self.device}...", file=sys.stderr)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, device=self.device
        )
        return _l2_normalize(np.asarray(emb, dtype=np.float32))

    def get_embedding(self, text: str) -> np.ndarray:
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
    device: Optional[str] = None,
) -> Embedder:
    """Get an embedder for the specified (or configured) backend.

    Args:
        backend: "fastembed" (default), "model2vec", "sentence-transformers", or "api"
        model: Model identifier (overrides env config)
        api_key: API key for 'api' backend (overrides env config)
        base_url: API base URL (overrides env config)
        dim: Embedding dimension (overrides env config)
        device: Device for torch-based backends (auto-detects if None)

    Returns:
        Embedder instance

    Raises:
        ValueError: If backend is unknown or required params are missing
    """
    import os

    config = env_config()

    # Explicit args win; gaps filled by env_config()
    backend = (backend or config.backend or "fastembed").lower()

    # Model default must be derived from the *effective* backend, not from the
    # EMBEDDING_BACKEND env var (which may differ when --backend is passed explicitly).
    _DEFAULT_MODELS = {
        "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }
    if model is None:
        model = os.getenv("EMBEDDING_MODEL") or _DEFAULT_MODELS.get(
            backend, config.model
        )

    if backend == "model2vec" and not model:
        raise ValueError(
            "model2vec requires --model (path to a distilled model, "
            "e.g. --model models/m2v-minilm)"
        )

    api_key = api_key or config.api_key
    base_url = base_url or config.base_url
    dim = dim if dim is not None else config.dim

    if backend == "fastembed":
        return FastEmbedEmbedder(model_name=model)
    elif backend == "model2vec":
        return Model2VecEmbedder(model_name=model)
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model, device=device)
    elif backend == "api":
        return ApiEmbedder(api_key=api_key, base_url=base_url, model=model, dim=dim)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND: {backend!r} "
            "(expected 'fastembed', 'model2vec', 'sentence-transformers', or 'api')"
        )


class Reranker(Protocol):
    """Protocol for reranking backends."""

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return

        Returns:
            List of dicts with 'index' and 'relevance_score', sorted by score desc
        """
        ...


class FastEmbedReranker:
    """Local reranking via fastembed TextCrossEncoder (ONNX/CPU)."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for local reranking. "
                "Install it with `pip install prussian-embeddings[local]`."
            ) from exc

        print(f"Loading reranker model: {model_name}...", file=sys.stderr)
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        scores = list(self.model.rerank(query, documents))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"index": i, "relevance_score": float(s)} for i, s in ranked]


class ApiReranker:
    """Remote reranking via EmbeddingClient (Jina API)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ):
        from .client import EmbeddingClient

        if not api_key:
            raise ValueError(
                "api_key is required for the 'api' reranker backend. "
                "Set via API_KEY or EMBEDDING_API_KEY environment variable."
            )

        self.client = EmbeddingClient(
            api_key=api_key,
            base_url=base_url,
            reranker_model=model,
        )

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        import anyio

        return anyio.run(self.client.rerank, query, documents, top_n)


def get_reranker(
    backend: Optional[str] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Reranker:
    """Get a reranker for the specified (or configured) backend.

    Args:
        backend: "fastembed" (default) or "api"
        model: Model identifier (overrides env config)
        api_key: API key for 'api' backend (overrides env config)
        base_url: API base URL (overrides env config)

    Returns:
        Reranker instance

    Raises:
        ValueError: If backend is unknown or required params are missing
    """
    config = env_config()

    backend = (backend or config.reranker_backend or "fastembed").lower()
    model = model or config.reranker_model

    if backend == "fastembed":
        return FastEmbedReranker(model_name=model)
    elif backend == "api":
        return ApiReranker(
            api_key=api_key or config.api_key,
            base_url=base_url or config.base_url,
            model=model,
        )
    else:
        raise ValueError(
            f"Unknown RERANKER_BACKEND: {backend!r} (expected 'fastembed' or 'api')"
        )
