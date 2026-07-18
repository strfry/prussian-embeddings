"""Embedding and reranking API client."""

from typing import Any, Dict, List

import httpx
import numpy as np

from .config import env_config


class EmbeddingClient:
    """Client for embedding and reranking APIs (e.g., Jina AI)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        embedding_model: str = "",
        embedding_dim: int = 0,
        reranker_model: str = "",
        timeout: float = 120.0,
    ):
        """Initialize embedding client.
        
        Args:
            api_key: API authentication key
            base_url: API endpoint base URL
            embedding_model: Model ID for embeddings
            embedding_dim: Embedding dimension
            reranker_model: Model ID for reranking
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.reranker_model = reranker_model
        self.timeout = timeout

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
        )

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim), dtype float32
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

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

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of shape (embedding_dim,), dtype float32
        """
        embeddings = self.get_embeddings([text])
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
