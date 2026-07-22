"""Embedding and reranking API client."""

from typing import Any, Dict, List

import httpx
import numpy as np

from .config import env_config, resolve_provider


class EmbeddingClient:
    """Client for embedding and reranking APIs (e.g., Jina AI, Voyage)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        embedding_model: str = "",
        embedding_dim: int = 0,
        reranker_model: str = "",
        provider: str = "",
        timeout: float = 120.0,
    ):
        """Initialize embedding client.

        Args:
            api_key: API authentication key
            base_url: API endpoint base URL
            embedding_model: Model ID for embeddings
            embedding_dim: Embedding dimension
            reranker_model: Model ID for reranking
            provider: API provider ("voyage", "jina", "generic"); "" auto-detects
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.reranker_model = reranker_model
        self.timeout = timeout
        self.provider = resolve_provider(self.base_url, embedding_model, provider)

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        """Create client from environment variables."""
        config = env_config()
        return cls(
            api_key=config.api_key,
            base_url=config.base_url,
            embedding_model=config.model,
            embedding_dim=config.dim,
            reranker_model=config.reranker_model,
            provider=config.provider,
        )

    def _input_type_param(self, is_query: bool) -> Dict[str, Any]:
        """Provider-specific query/passage parameter for the embeddings request.

        Modern APIs distinguish queries from passages via a request parameter
        (not a text prefix), and the parameter name/values differ per provider:

        - voyage → ``input_type``: "query" / "document"
        - jina   → ``task``: "retrieval.query" / "retrieval.passage"
        - generic → none (OpenAI-compatible endpoints reject unknown fields)
        """
        if self.provider == "voyage":
            return {"input_type": "query" if is_query else "document"}
        if self.provider == "jina":
            return {"task": "retrieval.query" if is_query else "retrieval.passage"}
        return {}

    def get_embeddings(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            is_query: True for search queries, False for passages/documents.
                Controls the provider-specific query/passage parameter.

        Returns:
            numpy array of shape (len(texts), embedding_dim), dtype float32
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.embedding_model,
            "input": texts,
        }
        payload.update(self._input_type_param(is_query))

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/embeddings", headers=headers, json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"Embedding API error: {response.status_code} - {response.text}"
                )

            try:
                data = response.json()
            except Exception:
                raise Exception(
                    f"Embedding API returned invalid JSON: {response.text[:200]}"
                )

            embeddings = []
            for item in data["data"]:
                vec = item["embedding"]
                if not vec or not any(
                    isinstance(v, (int, float)) and v != 0 for v in vec
                ):
                    raise Exception(
                        f"Embedding API returned empty/null embedding vector "
                        f"(len={len(vec) if vec else 0}). "
                        f"Check if the model '{self.embedding_model}' is loaded "
                        f"correctly on {self.base_url}"
                    )
                embeddings.append(vec)
            return np.array(embeddings, dtype=np.float32)

    def get_embedding(self, text: str, *, is_query: bool = False) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: Text to embed
            is_query: True for a search query, False for a passage/document.

        Returns:
            numpy array of shape (embedding_dim,), dtype float32
        """
        embeddings = self.get_embeddings([text], is_query=is_query)
        return embeddings[0]

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return
            return_documents: Whether to include full documents in response

        Returns:
            List of dicts with index, relevance_score, and optionally document
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/rerank", headers=headers, json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"Rerank API error: {response.status_code} - {response.text}"
                )

            body = response.json()
            results = body.get("data") or body.get("results") or []
            return results[:top_n]
