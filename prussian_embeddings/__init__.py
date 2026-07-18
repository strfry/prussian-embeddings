"""Embeddings for Prussian dictionary."""

from .backends import get_embedder, FastEmbedEmbedder, Model2VecEmbedder, ApiEmbedder
from .client import EmbeddingClient
from .store import EmbeddingStore
from .config import env_config, EnvConfig
from .passages import (
    has_translations,
    translations,
    description,
    word_type,
    make_passage,
    LANGUAGE_ORDER,
)

__all__ = [
    "get_embedder",
    "FastEmbedEmbedder",
    "Model2VecEmbedder",
    "ApiEmbedder",
    "EmbeddingClient",
    "EmbeddingStore",
    "env_config",
    "EnvConfig",
    "has_translations",
    "translations",
    "description",
    "word_type",
    "make_passage",
    "LANGUAGE_ORDER",
]
