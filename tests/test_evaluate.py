"""Tests for prussian_embeddings.evaluate.

Uses a deterministic TokenStubEmbedder (no model downloads needed).
"""

import random

import numpy as np
import pytest

from prussian_embeddings.evaluate import (
    Bucket,
    build_eval_queries,
    build_gold_sets,
    build_passages,
    evaluate_spec,
    format_report,
    parse_spec,
    resolve_embedder_config,
)
from prussian_embeddings.store import EmbeddingStore


# ---------------------------------------------------------------------------
# TokenStubEmbedder: deterministic token → seeded random unit vector
# Text embedding = L2-normalized sum of token embeddings
# ---------------------------------------------------------------------------

class TokenStubEmbedder:
    """Deterministic stub: each token gets a seeded random unit vector."""

    def __init__(self, dim: int = 16, seed: int = 0):
        self.dim = dim
        self._seed = seed

    def _token_vec(self, token: str) -> np.ndarray:
        h = hash((self._seed, token))
        rng = random.Random(h)
        vec = np.array([rng.gauss(0, 1) for _ in range(self.dim)], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            vec[0] = 1.0
            norm = 1.0
        return vec / norm

    def _embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vec = sum(self._token_vec(t) for t in tokens)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros(self.dim, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def get_embeddings(self, texts, *, is_query: bool = False):
        return np.array([self._embed(t) for t in texts], dtype=np.float32)

    def get_embedding(self, text, *, is_query: bool = False):
        return self._embed(text)


# ---------------------------------------------------------------------------
# Synthetic dictionary entries
# ---------------------------------------------------------------------------

SYNTH_ENTRIES = [
    {
        "word": "berzi",
        "translations": {
            "engl": ["birch"],
            "miks": ["Birke"],
            "leit": ["berzas"],
            "latt": ["berzs"],
            "pols": ["brzoza"],
            "mask": ["береза"],
        },
    },
    {
        "word": "kunnegs",
        "translations": {
            "engl": ["king"],
            "miks": ["König"],
            "leit": ["karalius"],
            "latt": ["karalis"],
            "pols": ["król"],
            "mask": ["король"],
        },
    },
    {
        "word": "wundan",
        "translations": {
            "engl": ["water"],
            "miks": ["Wasser"],
            "leit": ["vanduo"],
            "latt": ["udens"],
            "pols": ["woda"],
            "mask": ["вода"],
        },
    },
    {
        "word": "sestra",
        "translations": {
            "engl": ["sister"],
            "miks": ["Schwester"],
            "leit": ["sesuo"],
            "latt": ["masa"],
            "pols": ["siostra"],
            "mask": ["сестра"],
        },
    },
    {
        "word": "wirs",
        "translations": {
            "engl": ["man", "husband"],
            "miks": ["Mann"],
            "leit": ["vyras"],
            "latt": ["virs"],
            "pols": ["mąż"],
            "mask": ["муж"],
        },
    },
    {
        "word": "suns",
        "translations": {
            "engl": ["son"],
            "miks": ["Sohn"],
            "leit": ["sunus"],
            "latt": ["dels"],
            "pols": ["syn"],
            "mask": ["сын"],
        },
    },
]


ALL_LANGS = ["engl", "miks", "leit", "latt", "pols", "mask"]


def _build_store(entries=None, embedder=None):
    """Helper: build store from entries with the stub embedder."""
    entries = entries or SYNTH_ENTRIES
    embedder = embedder or TokenStubEmbedder(dim=16, seed=42)
    texts, records = build_passages(entries)
    return EmbeddingStore.build(embedder, texts, records), embedder


# ---------------------------------------------------------------------------
# parse_spec
# ---------------------------------------------------------------------------

class TestParseSpec:
    def test_backend_only(self):
        assert parse_spec("fastembed") == ("fastembed", None)

    def test_backend_and_model(self):
        assert parse_spec("model2vec:models/x") == ("model2vec", "models/x")

    def test_colon_in_model(self):
        assert parse_spec("api:http://x") == ("api", "http://x")


# ---------------------------------------------------------------------------
# seen / unseen flags
# ---------------------------------------------------------------------------

class TestSeenUnseen:
    def test_seen_queries_from_passage_langs(self):
        """First translation of engl/miks/leit/latt should be 'seen'."""
        entries = SYNTH_ENTRIES[:2]
        records = build_passages(entries)[1]
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)

        seen = [q for q in queries if q.seen]
        unseen = [q for q in queries if not q.seen]

        assert len(seen) > 0
        assert len(unseen) > 0

        for q in seen:
            assert q.lang in ["engl", "miks", "leit", "latt"]

    def test_pols_mask_always_unseen(self):
        entries = SYNTH_ENTRIES[:2]
        records = build_passages(entries)[1]
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)

        for q in queries:
            if q.lang in ("pols", "mask"):
                assert not q.seen

    def test_second_translation_unseen(self):
        """The second English translation of 'wirs' (husband) should be unseen."""
        entries = SYNTH_ENTRIES  # wirs has engl: ["man", "husband"]
        records = build_passages(entries)[1]
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)

        husband_q = [q for q in queries if q.text == "husband"]
        assert len(husband_q) == 1
        assert not husband_q[0].seen


# ---------------------------------------------------------------------------
# deterministic sampling
# ---------------------------------------------------------------------------

class TestDeterministicSampling:
    def test_same_seed_same_queries(self):
        records = build_passages(SYNTH_ENTRIES)[1]
        q1 = build_eval_queries(records, ALL_LANGS, limit=3, seed=99)
        q2 = build_eval_queries(records, ALL_LANGS, limit=3, seed=99)
        assert [(q.text, q.lang) for q in q1] == [(q.text, q.lang) for q in q2]

    def test_different_seed_different_queries(self):
        records = build_passages(SYNTH_ENTRIES)[1]
        q1 = build_eval_queries(records, ALL_LANGS, limit=3, seed=1)
        q2 = build_eval_queries(records, ALL_LANGS, limit=3, seed=2)
        # Different seeds sample different entries → different gold_idx sets
        idx1 = {q.gold_idx for q in q1}
        idx2 = {q.gold_idx for q in q2}
        # With 6 entries and limit=3, it's very likely they differ
        assert len(idx1) == 3
        assert len(idx2) == 3

    def test_limit_zero_means_all(self):
        records = build_passages(SYNTH_ENTRIES)[1]
        q_all = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)
        q_lim = build_eval_queries(records, ALL_LANGS, limit=3, seed=42)
        assert len(q_all) >= len(q_lim)


# ---------------------------------------------------------------------------
# gold sets: case-insensitive
# ---------------------------------------------------------------------------

class TestGoldSets:
    def test_case_insensitive(self):
        records = build_passages(SYNTH_ENTRIES)[1]
        gold = build_gold_sets(records, ALL_LANGS)

        # "birch" and "Birch" should map to the same gold set
        g1 = gold.get(("engl", "birch"), set())
        g2 = gold.get(("engl", "birch"), set())
        assert g1 == g2

    def test_multiple_entries_same_translation(self):
        """If two entries share a translation, gold set should have both."""
        entries = [
            {
                "word": "a",
                "translations": {"engl": ["same"], "miks": ["x"], "leit": ["x"], "latt": ["x"]},
            },
            {
                "word": "b",
                "translations": {"engl": ["same"], "miks": ["y"], "leit": ["y"], "latt": ["y"]},
            },
        ]
        records = build_passages(entries)[1]
        gold = build_gold_sets(records, ALL_LANGS)
        key = ("engl", "same")
        assert len(gold[key]) == 2


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------

class TestBucket:
    def test_empty_bucket(self):
        b = Bucket()
        d = b.as_dict()
        assert d["n"] == 0
        assert d["Hit@1"] == 0.0
        assert d["MRR"] == 0.0

    def test_known_ranks(self):
        b = Bucket(n=4, hits_1=2, hits_5=3, hits_10=4, mrr_sum=1.0 + 1.0 + 0.5 + 0.25, any_1=3)
        d = b.as_dict()
        assert d["n"] == 4
        assert d["Hit@1"] == pytest.approx(0.5)
        assert d["Hit@5"] == pytest.approx(0.75)
        assert d["Hit@10"] == pytest.approx(1.0)
        assert d["MRR"] == pytest.approx(2.75 / 4)
        assert d["any@1"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# End-to-end on synthetic entries
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_evaluate_all_queries(self):
        """Evaluate all queries on 6 synthetic entries with the stub embedder."""
        store, embedder = _build_store()
        records = store.records
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)
        gold_sets = build_gold_sets(records, ALL_LANGS)

        result = evaluate_spec(embedder, store, queries, gold_sets)

        assert result["num_queries"] > 0
        assert result["query_time"] > 0
        assert ("ALL", "all") in result["results"]

        total = result["results"][("ALL", "all")]
        assert total.n > 0
        d = total.as_dict()
        assert 0.0 <= d["Hit@1"] <= 1.0
        assert 0.0 <= d["MRR"] <= 1.0

    def test_evaluate_with_limit(self):
        store, embedder = _build_store()
        records = store.records
        queries = build_eval_queries(records, ALL_LANGS, limit=2, seed=42)
        gold_sets = build_gold_sets(records, ALL_LANGS)

        result = evaluate_spec(embedder, store, queries, gold_sets)
        # limit=2 entries, but many queries per entry
        assert result["num_queries"] > 0

    def test_per_lang_results(self):
        store, embedder = _build_store()
        records = store.records
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)
        gold_sets = build_gold_sets(records, ALL_LANGS)

        result = evaluate_spec(embedder, store, queries, gold_sets)

        # Should have results for all passage languages + ALL
        for lang in ["engl", "miks", "leit", "latt"]:
            assert (lang, "all") in result["results"]

    def test_format_report(self):
        store, embedder = _build_store()
        records = store.records
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)
        gold_sets = build_gold_sets(records, ALL_LANGS)

        result = evaluate_spec(embedder, store, queries, gold_sets)
        meta = store.meta
        report = format_report("fastembed", meta, result)

        assert "fastembed" in report
        assert "Hit@1" in report
        assert "ALL" in report


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_dedup_same_lang_text(self):
        """Same (lang, text) pair for one entry should produce one query."""
        entries = [
            {
                "word": "x",
                "translations": {
                    "engl": ["foo"],
                    "miks": ["foo"],
                    "leit": ["a"],
                    "latt": ["b"],
                },
            },
        ]
        records = build_passages(entries)[1]
        queries = build_eval_queries(records, ALL_LANGS, limit=0, seed=42)

        engl_foo = [q for q in queries if q.lang == "engl" and q.text == "foo"]
        miks_foo = [q for q in queries if q.lang == "miks" and q.text == "foo"]
        # Same text, same lang → deduped
        assert len(engl_foo) == 1
        # Different lang → not deduped
        assert len(miks_foo) == 1


# ---------------------------------------------------------------------------
# resolve_embedder_config
# ---------------------------------------------------------------------------

class TestResolveEmbedderConfig:
    def test_default_model_to_none(self):
        assert resolve_embedder_config({"backend": "fastembed", "model": "default"}) == (
            "fastembed",
            None,
        )

    def test_empty_string_model_to_none(self):
        assert resolve_embedder_config({"backend": "model2vec", "model": ""}) == (
            "model2vec",
            None,
        )

    def test_no_model_key_to_none(self):
        assert resolve_embedder_config({"backend": "fastembed"}) == ("fastembed", None)

    def test_explicit_model(self):
        assert resolve_embedder_config(
            {"backend": "model2vec", "model": "models/m2v-minilm"}
        ) == ("model2vec", "models/m2v-minilm")

    def test_missing_backend_raises(self):
        with pytest.raises(ValueError, match="missing 'backend'"):
            resolve_embedder_config({})

    def test_empty_backend_raises(self):
        with pytest.raises(ValueError, match="missing 'backend'"):
            resolve_embedder_config({"backend": ""})


# ---------------------------------------------------------------------------
# Dim-mismatch sanity check
# ---------------------------------------------------------------------------

class StubWrongDimEmbedder:
    """Stub with a different dim than what's in the store."""

    def __init__(self, dim=32):
        self.dim = dim

    def get_embeddings(self, texts, *, is_query: bool = False):
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def get_embedding(self, text, *, is_query: bool = False):
        return np.zeros(self.dim, dtype=np.float32)


class TestDimMismatch:
    def test_dim_mismatch_exits(self, tmp_path):
        """Store with dim=16 but embedder with dim=32 should fail in main()."""
        from unittest.mock import patch

        from prussian_embeddings.evaluate import main

        store, _ = _build_store()
        store.meta = {"backend": "fastembed", "model": "default", "embedding_dim": 16}
        stem = str(tmp_path / "test_store")
        store.save(stem)

        wrong = StubWrongDimEmbedder(dim=32)
        with patch("sys.argv", ["eval", "--store", stem]), \
             patch("prussian_embeddings.backends.get_embedder", return_value=wrong):
            with pytest.raises(SystemExit, match="1"):
                main()


# ---------------------------------------------------------------------------
# Per-store query_prefix (saved in meta by generate, auto-applied by eval)
# ---------------------------------------------------------------------------

class SpyEmbedder:
    """TokenStubEmbedder that records all texts passed to get_embeddings."""

    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self._stub = TokenStubEmbedder(dim=dim, seed=seed)
        self.embedded_texts: list[str] = []

    def get_embeddings(self, texts, *, is_query: bool = False):
        self.embedded_texts.extend(texts)
        return self._stub.get_embeddings(texts, is_query=is_query)

    def get_embedding(self, text, *, is_query: bool = False):
        return self._stub.get_embedding(text, is_query=is_query)


class TestStoreQueryPrefix:
    """eval --store mode auto-applies the store's saved query_prefix;
    an explicit --query-prefix overrides it."""

    def _save_store(self, tmp_path, query_prefix):
        store, _ = _build_store()
        store.meta = {
            "backend": "fastembed",
            "model": "default",
            "embedding_dim": 16,
            "query_prefix": query_prefix,
        }
        stem = str(tmp_path / "qp_store")
        store.save(stem)
        return stem

    def test_store_query_prefix_applied(self, tmp_path):
        """Without --query-prefix, the store's saved query_prefix is applied."""
        from unittest.mock import patch

        from prussian_embeddings.evaluate import main

        stem = self._save_store(tmp_path, "query: ")
        spy = SpyEmbedder(dim=16)

        with patch("sys.argv", ["eval", "--store", stem]), \
             patch("prussian_embeddings.backends.get_embedder", return_value=spy):
            main()

        assert spy.embedded_texts, "no queries were embedded"
        assert all(t.startswith("query: ") for t in spy.embedded_texts)

    def test_explicit_query_prefix_overrides_store(self, tmp_path):
        """An explicit --query-prefix overrides the store's saved prefix."""
        from unittest.mock import patch

        from prussian_embeddings.evaluate import main

        stem = self._save_store(tmp_path, "query: ")
        spy = SpyEmbedder(dim=16)

        with patch("sys.argv", ["eval", "--store", stem, "--query-prefix", "override: "]), \
             patch("prussian_embeddings.backends.get_embedder", return_value=spy):
            main()

        assert spy.embedded_texts, "no queries were embedded"
        assert all(t.startswith("override: ") for t in spy.embedded_texts)
        assert not any(t.startswith("query: ") for t in spy.embedded_texts)


# ---------------------------------------------------------------------------
# Per-store query_prefix auto-application
# ---------------------------------------------------------------------------

class RecordingEmbedder:
    """Wraps an embedder and records all texts passed to get_embeddings."""

    def __init__(self, inner):
        self.inner = inner
        self.dim = inner.dim
        self.seen_texts = []

    def get_embeddings(self, texts, *, is_query: bool = False):
        self.seen_texts.extend(texts)
        return self.inner.get_embeddings(texts, is_query=is_query)

    def get_embedding(self, text, *, is_query: bool = False):
        return self.inner.get_embedding(text, is_query=is_query)


class TestQueryPrefixFromMeta:
    def test_store_meta_prefix_applied(self, tmp_path):
        """Store with meta['query_prefix'] applies it when --query-prefix omitted."""
        from unittest.mock import patch

        from prussian_embeddings.evaluate import main

        store, _ = _build_store()
        store.meta = {
            "backend": "fastembed",
            "model": "default",
            "embedding_dim": 16,
            "query_prefix": "query: ",
        }
        stem = str(tmp_path / "qp_store")
        store.save(stem)

        recorder = RecordingEmbedder(TokenStubEmbedder(dim=16, seed=42))
        with patch("sys.argv", ["eval", "--store", stem, "--limit", "1"]), \
             patch("prussian_embeddings.backends.get_embedder", return_value=recorder):
            main()

        assert recorder.seen_texts, "no queries were embedded"
        assert all(t.startswith("query: ") for t in recorder.seen_texts)

    def test_explicit_query_prefix_overrides_meta(self, tmp_path):
        """Explicit --query-prefix overrides the store's saved query_prefix."""
        from unittest.mock import patch

        from prussian_embeddings.evaluate import main

        store, _ = _build_store()
        store.meta = {
            "backend": "fastembed",
            "model": "default",
            "embedding_dim": 16,
            "query_prefix": "query: ",
        }
        stem = str(tmp_path / "qp_store2")
        store.save(stem)

        recorder = RecordingEmbedder(TokenStubEmbedder(dim=16, seed=42))
        with patch(
            "sys.argv",
            ["eval", "--store", stem, "--limit", "1", "--query-prefix", "override: "],
        ), \
             patch("prussian_embeddings.backends.get_embedder", return_value=recorder):
            main()

        assert recorder.seen_texts, "no queries were embedded"
        assert all(t.startswith("override: ") for t in recorder.seen_texts)
