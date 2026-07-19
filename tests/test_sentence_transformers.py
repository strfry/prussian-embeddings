"""Tests for sentence-transformers backend (no model downloads)."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prussian_embeddings.backends import _pick_device, get_embedder


# ---------------------------------------------------------------------------
# _pick_device
# ---------------------------------------------------------------------------


class TestPickDevice:
    def test_explicit_device_passthrough(self):
        assert _pick_device("cuda") == "cuda"
        assert _pick_device("xpu") == "xpu"
        assert _pick_device("cpu") == "cpu"

    def test_auto_prefers_xpu(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.xpu = types.SimpleNamespace(is_available=lambda: True)
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert _pick_device(None) == "xpu"
            assert _pick_device("auto") == "xpu"

    def test_auto_falls_back_to_cuda(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.xpu = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert _pick_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.xpu = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert _pick_device("auto") == "cpu"

    def test_no_torch_falls_back_to_cpu(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _pick_device(None) == "cpu"


# ---------------------------------------------------------------------------
# get_embedder routing
# ---------------------------------------------------------------------------


class TestGetEmbedderRouting:
    def test_sentence_transformers_routes_correctly(self):
        """get_embedder('sentence-transformers') should create SentenceTransformerEmbedder."""
        stub_cls = MagicMock()
        stub_instance = MagicMock()
        stub_instance.get_sentence_embedding_dimension.return_value = 384
        stub_cls.return_value = stub_instance

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = stub_cls

        with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
            embedder = get_embedder("sentence-transformers", model="some-model", device="cpu")

        stub_cls.assert_called_once_with("some-model", device="cpu")
        assert embedder.dim == 384

    def test_sentence_transformers_device_passthrough(self):
        """Device arg should be forwarded to SentenceTransformer."""
        stub_cls = MagicMock()
        stub_instance = MagicMock()
        stub_instance.get_sentence_embedding_dimension.return_value = 128
        stub_cls.return_value = stub_instance

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = stub_cls

        with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
            embedder = get_embedder("sentence-transformers", device="xpu")

        assert stub_cls.call_args[1]["device"] == "xpu"

    def test_fastembed_unchanged(self):
        """fastembed backend should still work unchanged."""
        with patch("prussian_embeddings.backends.FastEmbedEmbedder") as mock_fe:
            get_embedder("fastembed")
            mock_fe.assert_called_once()

    def test_model2vec_unchanged(self):
        """model2vec backend should still work unchanged."""
        with patch("prussian_embeddings.backends.Model2VecEmbedder") as mock_m2v:
            get_embedder("model2vec", model="some/path")
            mock_m2v.assert_called_once_with(model_name="some/path")

    def test_api_unchanged(self):
        """api backend should still work unchanged."""
        with patch("prussian_embeddings.backends.ApiEmbedder") as mock_api:
            get_embedder("api", api_key="key", base_url="url", model="m", dim=64)
            mock_api.assert_called_once_with(api_key="key", base_url="url", model="m", dim=64)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown EMBEDDING_BACKEND"):
            get_embedder("nonexistent")
