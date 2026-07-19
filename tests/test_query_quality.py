"""Cross-backend query quality tests.

These tests verify that querying for a known translation returns the correct
Prussian entry as the top result. They test the full pipeline:
    build passage → embed corpus → embed query → cosine similarity → top-k
"""

from pathlib import Path

import pytest

from prussian_embeddings.store import EmbeddingStore
from prussian_embeddings.passages import make_passage

# ---------------------------------------------------------------------------
# Synthetic Prussian dictionary entries (word, translations)
# ---------------------------------------------------------------------------
ENTRIES = [
    {"word": "berzi",   "translations": {"engl": ["birch"],   "miks": ["Birke"],     "leit": ["beržas"], "latt": ["bērzs"]}},
    {"word": "kunnegs", "translations": {"engl": ["king"],    "miks": ["König"],     "leit": ["karalius"], "latt": ["karalis"]}},
    {"word": "wundan",  "translations": {"engl": ["water"],   "miks": ["Wasser"],    "leit": ["vanduo"],   "latt": ["ūdens"]}},
    {"word": "sestrā",  "translations": {"engl": ["sister"],  "miks": ["Schwester"], "leit": ["sesuo"],    "latt": ["māsa"]}},
    {"word": "wīrs",    "translations": {"engl": ["man", "husband"], "miks": ["Mann"], "leit": ["vyras"], "latt": ["vīrs"]}},
    {"word": "sūns",    "translations": {"engl": ["son"],     "miks": ["Sohn"],      "leit": ["sūnus"],   "latt": ["dēls"]}},
    {"word": "ackis",   "translations": {"engl": ["eye"],     "miks": ["Auge"],      "leit": ["akis"],    "latt": ["acs"]}},
    {"word": "ausis",   "translations": {"engl": ["ear"],     "miks": ["Ohr"],       "leit": ["ausis"],   "latt": ["auss"]}},
]

QUERY_CASES = [
    ("Birke",     "berzi"),
    ("König",     "kunnegs"),
    ("Wasser",    "wundan"),
    ("Schwester", "sestrā"),
]


def _build_store(embedder) -> EmbeddingStore:
    texts, records = [], []
    for entry in ENTRIES:
        passage = make_passage(entry, include_prussian=True,
                               langs=["engl", "miks", "leit", "latt"])
        if passage:
            texts.append(passage)
            records.append(entry)
    return EmbeddingStore.build(embedder, texts, records)


def _top1_word(store, embedder, query: str) -> str:
    results = store.query(embedder, query, k=1)
    assert results, f"No results for query {query!r}"
    record, _score = results[0]
    return record["word"]


def _in_top_k(store, embedder, query: str, k: int = 3) -> list[str]:
    results = store.query(embedder, query, k=k)
    return [r["word"] for r, _ in results]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fastembed_embedder():
    pytest.importorskip("fastembed", reason="fastembed not installed")
    from prussian_embeddings import FastEmbedEmbedder
    return FastEmbedEmbedder()


@pytest.fixture(scope="module")
def distilled_embedder():
    """Load the true model2vec embedder (distilled model). Skips if not present."""
    import glob as globmod
    models_dir = Path(__file__).parent.parent / "models"
    matches = globmod.glob(str(models_dir / "*/model.safetensors"))
    if not matches:
        pytest.skip("No distilled model found in models/")
    model_path = Path(matches[0]).parent
    pytest.importorskip("model2vec", reason="model2vec not installed")
    from prussian_embeddings import Model2VecEmbedder
    return Model2VecEmbedder(str(model_path))


# ---------------------------------------------------------------------------
# Unit: synthetic corpus, consistent empty prefix (both corpus and query)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_word", QUERY_CASES)
def test_fastembed_retrieval(fastembed_embedder, query, expected_word):
    store = _build_store(fastembed_embedder)
    result = _top1_word(store, fastembed_embedder, query)
    assert result == expected_word, f"Query {query!r}: expected {expected_word!r}, got {result!r}"


@pytest.mark.parametrize("query,expected_word", QUERY_CASES)
def test_distilled_retrieval(distilled_embedder, query, expected_word):
    store = _build_store(distilled_embedder)
    result = _top1_word(store, distilled_embedder, query)
    assert result == expected_word, f"Query {query!r}: expected {expected_word!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Integration: query real precomputed artifact (skipped if not present)
# ---------------------------------------------------------------------------

ARTIFACT_CASES = [
    pytest.param(
        "Birke", "berzi",
        marks=pytest.mark.xfail(
            reason="fastembed >=0.6 switched MiniLM to mean pooling; "
            "'Birke' drops to rank ~16 on the full corpus",
            strict=False,
        ),
    ),
    ("König",     "kunnegs"),
    ("Wasser",    "wundan"),
    ("Schwester", "sestrā"),
]


def _load_artifact(name: str) -> EmbeddingStore:
    stem = Path(__file__).parent.parent / "data" / name
    if not (Path(str(stem) + ".embeddings.npy")).exists():
        pytest.skip(f"Artifact {name} not present")
    return EmbeddingStore.load(str(stem))


@pytest.mark.parametrize("query,expected_word", ARTIFACT_CASES)
def test_fastembed_artifact_retrieval(fastembed_embedder, query, expected_word):
    store = _load_artifact("embeddings_fastembed")
    top3 = _in_top_k(store, fastembed_embedder, query, k=3)
    assert expected_word in top3, f"Query {query!r}: expected {expected_word!r} in top-3, got {top3}"
