"""Tests for provider resolution and query/passage request parameters.

Modern embedding APIs distinguish search queries from indexed passages via a
request *parameter* (Voyage: ``input_type``, Jina: ``task``) rather than a text
prefix. These tests pin the parameter name/values per provider and verify the
``is_query`` flag propagates from the query call sites down into the request.
"""

import numpy as np
import pytest

from prussian_embeddings.client import EmbeddingClient
from prussian_embeddings.config import resolve_provider


# ── Provider resolution ────────────────────────────────────────────────────


class TestResolveProvider:
    def test_explicit_wins(self):
        assert resolve_provider("https://api.jina.ai", "voyage-3", "voyage") == "voyage"

    def test_detect_jina_from_url(self):
        assert resolve_provider("https://api.jina.ai/v1", "some-model") == "jina"

    def test_detect_voyage_from_url(self):
        assert resolve_provider("https://api.voyageai.com/v1", "m") == "voyage"

    def test_detect_voyage_from_model_slug(self):
        # LiteLLM-style provider/model slug, generic base URL
        assert resolve_provider("https://gateway.local", "voyage/voyage-3") == "voyage"

    def test_detect_jina_from_model_slug(self):
        assert resolve_provider("https://gateway.local", "jina_ai/jina-embeddings-v3") == "jina"

    def test_generic_fallback(self):
        assert resolve_provider("https://api.openai.com/v1", "text-embedding-3-small") == "generic"


# ── Query/passage parameter per provider ───────────────────────────────────


class TestInputTypeParam:
    @pytest.mark.parametrize(
        "provider,is_query,expected",
        [
            ("voyage", True, {"input_type": "query"}),
            ("voyage", False, {"input_type": "document"}),
            ("jina", True, {"task": "retrieval.query"}),
            ("jina", False, {"task": "retrieval.passage"}),
            ("generic", True, {}),
            ("generic", False, {}),
        ],
    )
    def test_param(self, provider, is_query, expected):
        client = EmbeddingClient(base_url="https://x", embedding_model="m", provider=provider)
        assert client._input_type_param(is_query) == expected


# ── End-to-end payload construction (httpx mocked) ─────────────────────────


class _FakeResponse:
    status_code = 200

    def __init__(self, dim: int, n: int):
        self._dim = dim
        self._n = n

    def json(self):
        vec = [0.1] * self._dim
        return {"data": [{"embedding": vec} for _ in range(self._n)]}


class _FakeHttpxClient:
    """Captures the JSON payload of the last POST."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        _FakeHttpxClient.captured = json
        return _FakeResponse(dim=3, n=len(json["input"]))


@pytest.fixture
def mock_httpx(monkeypatch):
    import prussian_embeddings.client as client_mod

    _FakeHttpxClient.captured = {}
    monkeypatch.setattr(client_mod.httpx, "Client", _FakeHttpxClient)
    return _FakeHttpxClient


def test_jina_query_payload(mock_httpx):
    client = EmbeddingClient(base_url="https://api.jina.ai", embedding_model="jina-embeddings-v3")
    client.get_embeddings(["wasser"], is_query=True)
    assert mock_httpx.captured["task"] == "retrieval.query"


def test_jina_passage_payload(mock_httpx):
    client = EmbeddingClient(base_url="https://api.jina.ai", embedding_model="jina-embeddings-v3")
    client.get_embeddings(["deiws: god"], is_query=False)
    assert mock_httpx.captured["task"] == "retrieval.passage"


def test_voyage_query_payload(mock_httpx):
    client = EmbeddingClient(base_url="https://api.voyageai.com", embedding_model="voyage-3")
    client.get_embeddings(["wasser"], is_query=True)
    assert mock_httpx.captured["input_type"] == "query"


def test_voyage_passage_payload(mock_httpx):
    client = EmbeddingClient(base_url="https://api.voyageai.com", embedding_model="voyage-3")
    client.get_embeddings(["deiws: god"], is_query=False)
    assert mock_httpx.captured["input_type"] == "document"


def test_generic_no_extra_param(mock_httpx):
    client = EmbeddingClient(base_url="https://api.openai.com", embedding_model="text-embedding-3-small")
    client.get_embeddings(["wasser"], is_query=True)
    assert "task" not in mock_httpx.captured
    assert "input_type" not in mock_httpx.captured


def test_get_embedding_single_passes_is_query(mock_httpx):
    client = EmbeddingClient(base_url="https://api.jina.ai", embedding_model="jina-embeddings-v3")
    client.get_embedding("wasser", is_query=True)
    assert mock_httpx.captured["task"] == "retrieval.query"


# ── is_query propagation from the query call sites ─────────────────────────


class _SpyEmbedder:
    """Records the is_query flag seen on the most recent call."""

    dim = 4

    def __init__(self):
        self.last_is_query = None

    def get_embeddings(self, texts, *, is_query: bool = False):
        self.last_is_query = is_query
        v = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))
        return v

    def get_embedding(self, text, *, is_query: bool = False):
        self.last_is_query = is_query
        return self.get_embeddings([text], is_query=is_query)[0]


def test_store_query_marks_is_query_true():
    from prussian_embeddings.store import EmbeddingStore

    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store = EmbeddingStore(embeddings=emb, records=[{"word": "deiws"}])
    spy = _SpyEmbedder()
    store.query(spy, "god", k=1)
    assert spy.last_is_query is True


def test_hybrid_query_marks_is_query_true():
    from prussian_embeddings.hybrid import BM25Index, hybrid_query
    from prussian_embeddings.store import EmbeddingStore

    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store = EmbeddingStore(embeddings=emb, records=[{"text": "deiws god"}])
    bm25 = BM25Index(["deiws god"])
    spy = _SpyEmbedder()
    hybrid_query(store, spy, bm25, "god", k=1)
    assert spy.last_is_query is True
