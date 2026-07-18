"""Environment configuration for embeddings."""

import os
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration from environment variables."""

    backend: str
    model: str
    dim: int
    reranker_model: str
    api_key: str
    base_url: str


def env_config() -> EnvConfig:
    """Read environment configuration for embeddings.
    
    Reads from env vars (never at import time, only on call):
    - EMBEDDING_BACKEND: "fastembed" (default), "model2vec", or "api"
    - EMBEDDING_MODEL: model identifier
    - EMBEDDING_DIM: embedding dimension (default depends on backend)
    - RERANKER_MODEL: reranker model for API backend
    - API_KEY / JINA_API_KEY: API authentication (JINA_API_KEY is fallback)
    - API_BASE_URL / JINA_BASE_URL: API endpoint (JINA_BASE_URL defaults to https://api.jina.ai)
    """
    backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()
    
    # Model defaults depend on backend
    if backend == "fastembed":
        default_model = "intfloat/multilingual-e5-small"
        default_dim = 384
    elif backend == "model2vec":
        default_model = "minishlab/potion-multilingual-128M"
        default_dim = 128
    else:  # api
        default_model = "jina-embeddings-v5-text-small"
        default_dim = 1024
    
    model = os.getenv("EMBEDDING_MODEL", default_model)
    dim = int(os.getenv("EMBEDDING_DIM", str(default_dim)))
    
    reranker_model = os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
    
    # API key: try API_KEY first, fall back to JINA_API_KEY
    api_key = os.getenv("API_KEY", "") or os.getenv("JINA_API_KEY", "")
    
    # API base URL: try API_BASE_URL first, fall back to JINA_BASE_URL or default
    base_url = (
        os.getenv("API_BASE_URL", "")
        or os.getenv("JINA_BASE_URL", "https://api.jina.ai")
    )
    
    return EnvConfig(
        backend=backend,
        model=model,
        dim=dim,
        reranker_model=reranker_model,
        api_key=api_key,
        base_url=base_url,
    )
