"""Tests for prussian_embeddings.vocabulary."""

import tempfile
from pathlib import Path

from prussian_embeddings.vocabulary import (
    build_vocabulary,
    load_vocabulary,
    save_vocabulary,
)

ENTRIES = [
    {
        "word": "berzi",
        "translations": {
            "engl": ["birch"],
            "miks": ["Birke"],
            "leit": ["beržas"],
            "latt": ["bērzs"],
            "pols": ["brzoza"],
            "mask": ["берёза"],
        },
    },
    {
        "word": "kunnegs",
        "translations": {
            "engl": ["king", "King"],
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
            "miks": ["weißer Ton"],
            "leit": ["vanduo"],
            "latt": ["ūdens"],
        },
    },
]


def test_basic():
    vocab = build_vocabulary(ENTRIES)
    assert "birch" in vocab
    assert "birke" in vocab  # lowercased
    assert "beržas" in vocab
    assert vocab == sorted(vocab)


def test_deduplication():
    vocab = build_vocabulary(ENTRIES)
    assert vocab == sorted(set(vocab))


def test_lowercasing():
    vocab_lower = build_vocabulary(ENTRIES, lowercase=True)
    vocab_raw = build_vocabulary(ENTRIES, lowercase=False)
    assert "king" in vocab_lower
    assert "King" in vocab_raw
    assert "King" not in vocab_lower


def test_multiword_translation():
    """'weißer Ton' should produce two separate tokens."""
    vocab = build_vocabulary(ENTRIES)
    assert "weißer" in vocab
    assert "ton" in vocab


def test_cyrillic():
    vocab = build_vocabulary(ENTRIES)
    assert "берёза" in vocab
    assert "король" in vocab


def test_empty_translations():
    entries = [{"word": "empty", "translations": {}}]
    vocab = build_vocabulary(entries)
    assert vocab == []


def test_lang_filter():
    vocab_en = build_vocabulary(ENTRIES, langs=["engl"])
    assert "birch" in vocab_en
    assert "birke" not in vocab_en


def test_roundtrip_save_load():
    vocab = build_vocabulary(ENTRIES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.txt"
        save_vocabulary(vocab, path)
        loaded = load_vocabulary(path)
    assert loaded == vocab


def test_load_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.txt"
        path.write_text("", encoding="utf-8")
        loaded = load_vocabulary(path)
    assert loaded == []
