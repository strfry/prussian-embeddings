"""Environment configuration for embeddings."""

import os
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration from environment variables."""

    backend: str
    model: str
    dim: int
    reranker_backend: str
    reranker_model: str
    api_key: str
    base_url: str
    provider: str = ""
    device: str = ""


def resolve_provider(base_url: str = "", model: str = "", explicit: str = "") -> str:
    """Resolve the embedding API provider (for query/passage parameter selection).

    Modern embedding APIs signal query-vs-passage via a request *parameter*
    (Voyage: ``input_type``, Jina: ``task``) rather than a text prefix. Which
    parameter/values to use depends on the provider, resolved here.

    Precedence: explicit value wins; otherwise auto-detected from the base URL
    *or* the model slug (covers both direct provider APIs and LiteLLM-style
    ``provider/model`` slugs). Falls back to ``"generic"`` (no extra parameter,
    safe for plain OpenAI-compatible endpoints).

    Args:
        base_url: API endpoint base URL.
        model: Model identifier (may be a ``voyage/…`` / ``jina_ai/…`` slug).
        explicit: Explicit provider override (e.g. from EMBEDDING_PROVIDER).

    Returns:
        One of ``"voyage"``, ``"jina"``, or ``"generic"``.
    """
    if explicit:
        return explicit.lower()
    haystack = f"{base_url} {model}".lower()
    if "voyage" in haystack:
        return "voyage"
    if "jina" in haystack:
        return "jina"
    return "generic"


def env_config() -> EnvConfig:
    """Read environment configuration for embeddings.

    Reads from env vars (never at import time, only on call):
    - EMBEDDING_BACKEND: "fastembed" (default), "model2vec", "sentence-transformers", or "api"
    - EMBEDDING_MODEL: model identifier
    - EMBEDDING_DIM: embedding dimension (default depends on backend)
    - EMBEDDING_DEVICE: device for torch-based backends (default "" = auto)
    - EMBEDDING_PROVIDER: API provider for query/passage parameter selection
      ("voyage", "jina", "generic"; default "" = auto-detect from URL/model)
    - RERANKER_BACKEND: "fastembed" (default) or "api"
    - RERANKER_MODEL: reranker model
    - API_KEY / EMBEDDING_API_KEY: API authentication (JINA_API_KEY is fallback)
    - API_BASE_URL / EMBEDDING_BASE_URL: API endpoint (JINA_BASE_URL defaults to https://api.jina.ai)
    """
    backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()

    # Model defaults depend on backend
    if backend == "fastembed":
        default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        default_dim = 384
    elif backend == "model2vec":
        default_model = ""
        default_dim = 0
    elif backend == "sentence-transformers":
        default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        default_dim = 384
    else:  # api
        default_model = "jina-embeddings-v5-text-small"
        default_dim = 1024

    model = os.getenv("EMBEDDING_MODEL", default_model)
    dim = int(os.getenv("EMBEDDING_DIM", str(default_dim)))
    device = os.getenv("EMBEDDING_DEVICE", "")
    provider = os.getenv("EMBEDDING_PROVIDER", "")

    reranker_backend = os.getenv("RERANKER_BACKEND", "fastembed").lower()
    default_reranker_model = (
        "Xenova/ms-marco-MiniLM-L-6-v2"
        if reranker_backend == "fastembed"
        else "jina-reranker-v2-base-multilingual"
    )
    reranker_model = os.getenv("RERANKER_MODEL", default_reranker_model)

    # API key: try API_KEY first, fall back to JINA_API_KEY
    api_key = os.getenv("API_KEY", "") or os.getenv("EMBEDDING_API_KEY", "")

    # API base URL: try API_BASE_URL first, fall back to JINA_BASE_URL or default
    base_url = os.getenv("API_BASE_URL", "") or os.getenv(
        "EMBEDDING_BASE_URL", "https://api.jina.ai"
    )

    return EnvConfig(
        backend=backend,
        model=model,
        dim=dim,
        reranker_backend=reranker_backend,
        reranker_model=reranker_model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        device=device,
    )
