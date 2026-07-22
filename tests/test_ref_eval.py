"""Tests for ref-clustered rank evaluation (ref_eval.py)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from prussian_embeddings.evaluate import Bucket
from prussian_embeddings.ref_eval import (
    build_baseline_word_to_indices,
    build_eval_queries,
    compute_baseline_gold_indices,
    compute_chunk_gold_indices,
    load_chunks_jsonl,
    load_dictionary,
    split_queries_by_group,
)
from prussian_embeddings.store import EmbeddingStore


# ── Fixtures ──


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def chunks_jsonl(temp_dir):
    """Create a small chunks.jsonl for testing."""
    # Cluster 1: [lemma1, entry1, entry2] - one has translations
    # Cluster 2: [lemma2, entry3, entry4] - one has translations
    chunks = [
        {"lemma": "lemma0", "members": ["entry0"], "pos": "noun", "text": "entry0: x | y"},
        {
            "lemma": "lemma1",
            "members": ["entry1", "entry2"],
            "pos": "verb",
            "text": "entry1,entry2: a | b",
        },
        {
            "lemma": "lemma2",
            "members": ["entry3", "entry4"],
            "pos": "noun",
            "text": "entry3,entry4: c | d",
        },
    ]
    chunks_file = temp_dir / "chunks.jsonl"
    with open(chunks_file, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return str(chunks_file)


@pytest.fixture
def dictionary(temp_dir):
    """Create a small dictionary for testing."""
    entries = [
        {"word": "entry0", "translations": {"engl": ["zero"]}},  # has trans
        {"word": "entry1", "translations": {"engl": ["one"]}},  # has trans
        {"word": "entry2", "translations": {}},  # no translations
        {"word": "entry3", "translations": {"engl": ["three"]}},  # has trans
        {"word": "entry4", "translations": {}},  # no translations
    ]
    dict_file = temp_dir / "dict.json"
    with open(dict_file, "w") as f:
        json.dump(entries, f)
    return str(dict_file)


@pytest.fixture
def chunk_store_fixture(temp_dir):
    """Create a mock chunk store."""
    stem = str(temp_dir / "chunk_store")

    # 3 chunks, 10-dimensional embeddings (not realistic, just for testing)
    embeddings = np.random.randn(3, 10).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

    records = [
        {"word": "entry1"},
        {"word": "entry2"},
        {"word": "entry3"},
    ]

    meta = {
        "backend": "sentence-transformers",
        "model": "test-model",
        "embedding_dim": 10,
        "query_prefix": "query: ",
        "strategy": "chunks",
    }

    store = EmbeddingStore(embeddings, records, meta)
    store.save(stem)
    return stem


@pytest.fixture
def baseline_store_fixture(temp_dir):
    """Create a mock baseline store."""
    stem = str(temp_dir / "baseline_store")

    # 5 records, 10-dimensional embeddings
    embeddings = np.random.randn(5, 10).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

    records = [
        {"word": "entry0"},
        {"word": "entry1"},
        {"word": "entry2"},
        {"word": "entry3"},
        {"word": "entry4"},
    ]

    meta = {
        "backend": "sentence-transformers",
        "model": "test-model",
        "embedding_dim": 10,
        "query_prefix": "query: ",
        "strategy": "translations_only",
    }

    store = EmbeddingStore(embeddings, records, meta)
    store.save(stem)
    return stem


# ── Tests ──


def test_load_chunks_jsonl(chunks_jsonl):
    """Test loading chunks from JSONL."""
    chunks, lemma_to_members = load_chunks_jsonl(chunks_jsonl)

    assert len(chunks) == 3
    assert chunks[1]["lemma"] == "lemma1"
    assert chunks[1]["members"] == ["entry1", "entry2"]

    assert lemma_to_members["lemma1"] == {"entry1", "entry2"}
    assert lemma_to_members["lemma2"] == {"entry3", "entry4"}


def test_load_dictionary(dictionary):
    """Test loading dictionary."""
    dict_entries = load_dictionary(dictionary)

    assert len(dict_entries) == 5
    assert dict_entries["entry0"]["word"] == "entry0"
    assert dict_entries["entry1"]["translations"]["engl"] == ["one"]
    assert dict_entries["entry2"]["translations"] == {}


def test_load_dictionary_homographs(temp_dir):
    """Homographs: an entry with translations wins over one without."""
    entries = [
        {"word": "homo", "translations": {"engl": ["first"]}},
        {"word": "homo", "translations": {}},
        {"word": "other", "translations": {}},
        {"word": "other", "translations": {"engl": ["late"]}},
    ]
    dict_file = temp_dir / "homo.json"
    with open(dict_file, "w") as f:
        json.dump(entries, f)

    dict_entries = load_dictionary(str(dict_file))
    assert dict_entries["homo"]["translations"] == {"engl": ["first"]}
    assert dict_entries["other"]["translations"] == {"engl": ["late"]}


def test_build_eval_queries(chunks_jsonl, dictionary):
    """Test building evaluation queries.

    Only multi-member clusters with at least one translated member should be included.
    Clusters: 0 (single member), 1 (multi, has trans), 2 (multi, has trans).
    """
    chunks, _ = load_chunks_jsonl(chunks_jsonl)
    dict_entries = load_dictionary(dictionary)

    queries, excluded = build_eval_queries(chunks, dict_entries)

    # Cluster 0: single member, not included
    # Cluster 1: entry1 (has trans), entry2 (no trans) -> both in queries
    # Cluster 2: entry3 (has trans), entry4 (no trans) -> both in queries
    assert len(queries) == 4  # entry1, entry2, entry3, entry4
    assert excluded == 0  # No clusters excluded

    # Check that cluster 0 (single member) was skipped
    query_words = {q.word for q in queries}
    assert "entry0" not in query_words


def test_split_queries_by_group(chunks_jsonl, dictionary):
    """Test splitting queries into groups A and B."""
    chunks, _ = load_chunks_jsonl(chunks_jsonl)
    dict_entries = load_dictionary(dictionary)

    queries, _ = build_eval_queries(chunks, dict_entries)
    group_a, group_b = split_queries_by_group(queries, dict_entries)

    # Group A: entry1, entry3 (have translations)
    # Group B: entry2, entry4 (no translations)
    assert len(group_a) == 2
    assert len(group_b) == 2

    group_a_words = {q.word for q in group_a}
    group_b_words = {q.word for q in group_b}

    assert group_a_words == {"entry1", "entry3"}
    assert group_b_words == {"entry2", "entry4"}


def test_compute_chunk_gold_indices():
    """Test computing chunk gold indices."""
    from prussian_embeddings.ref_eval import RefEvalQuery

    query = RefEvalQuery(
        word="entry1",
        entry_idx=1,
        cluster_lemma="lemma1",
        cluster_members={"entry1", "entry2"},
    )

    gold = compute_chunk_gold_indices(query, query.entry_idx)
    assert gold == {1}


def test_compute_baseline_gold_indices(baseline_store_fixture):
    """Test computing baseline gold indices."""
    from prussian_embeddings.ref_eval import RefEvalQuery

    baseline_store = EmbeddingStore.load(baseline_store_fixture)
    word_to_indices = build_baseline_word_to_indices(baseline_store)

    query = RefEvalQuery(
        word="entry1",
        entry_idx=1,
        cluster_lemma="lemma1",
        cluster_members={"entry1", "entry2"},
    )

    gold = compute_baseline_gold_indices(query, baseline_store, word_to_indices)
    # Baseline has entry1 at idx 1 and entry2 at idx 2
    assert gold == {1, 2}


def test_build_baseline_word_to_indices(baseline_store_fixture):
    """Test building word-to-indices map from baseline."""
    baseline_store = EmbeddingStore.load(baseline_store_fixture)
    word_to_indices = build_baseline_word_to_indices(baseline_store)

    assert word_to_indices["entry0"] == [0]
    assert word_to_indices["entry1"] == [1]
    assert word_to_indices["entry4"] == [4]


def test_bucket_as_dict():
    """Test Bucket.as_dict()."""
    b = Bucket(n=10, hits_1=5, hits_5=8, hits_10=9, mrr_sum=7.5)
    d = b.as_dict()

    assert d["n"] == 10
    assert d["Hit@1"] == 0.5
    assert d["Hit@5"] == 0.8
    assert d["Hit@10"] == 0.9
    assert abs(d["MRR"] - 0.75) < 0.01


def test_bucket_empty():
    """Test Bucket with zero entries."""
    b = Bucket()
    d = b.as_dict()

    assert d["n"] == 0
    assert d["Hit@1"] == 0.0
    assert d["Hit@5"] == 0.0
    assert d["Hit@10"] == 0.0
    assert d["MRR"] == 0.0
