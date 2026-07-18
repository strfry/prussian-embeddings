"""Tests for passage formatting."""

import pytest

from prussian_embeddings.passages import (
    has_translations,
    translations,
    description,
    word_type,
    make_passage,
    LANGUAGE_ORDER,
)


def test_has_translations():
    """Test translation detection."""
    # Entry with translations
    assert has_translations({
        "word": "buttan",
        "translations": {
            "engl": ["button"],
            "miks": ["Knopf"],
        }
    }) is True
    
    # Entry without translations
    assert has_translations({
        "word": "test",
        "translations": {}
    }) is False
    
    # Entry with empty translation lists
    assert has_translations({
        "word": "test",
        "translations": {"engl": [], "miks": []}
    }) is False


def test_translations():
    """Test extracting translations."""
    entry = {
        "word": "buttan",
        "translations": {
            "engl": ["button", "knob"],
            "miks": ["Knopf"],
            "leit": ["mygtukas"],
            "latt": [],  # empty
            "pols": ["guzik"],
            "mask": ["кнопка"],
        }
    }
    
    trans = translations(entry, langs=["engl", "miks", "leit", "latt", "pols"])
    assert trans == ["button", "Knopf", "mygtukas", "guzik"]


def test_translations_partial():
    """Test with partial language list."""
    entry = {
        "translations": {
            "engl": ["hello"],
            "miks": ["hallo"],
        }
    }
    
    trans = translations(entry, langs=["engl", "miks", "leit"])
    assert trans == ["hello", "hallo"]  # leit missing


def test_description():
    """Test description extraction."""
    assert description({"desc": "[Advent MK]"}) == "[Advent MK]"
    assert description({}) == ""
    assert description({"desc": ""}) == ""


def test_word_type():
    """Test word type extraction."""
    assert word_type({"desc": "verb transitive"}) == "verb"
    assert word_type({"desc": "  noun"}) == "noun"
    assert word_type({"desc": ""}) == ""
    assert word_type({}) == ""
    assert word_type({"desc": "123abc"}) == "123abc"


def test_make_passage_with_prussian():
    """Test passage formatting with Prussian headword."""
    entry = {
        "word": "buttan",
        "translations": {
            "engl": ["button"],
            "miks": ["Knopf"],
            "leit": ["mygtukas"],
            "latt": ["poga"],
        }
    }
    
    passage = make_passage(
        entry,
        include_prussian=True,
        langs=["engl", "miks", "leit", "latt"],
        prefix="Document: "
    )
    assert passage == "Document: buttan: button | Knopf | mygtukas | poga"


def test_make_passage_without_prussian():
    """Test passage formatting without Prussian headword."""
    entry = {
        "word": "buttan",
        "translations": {
            "engl": ["button"],
            "miks": ["Knopf"],
            "leit": ["mygtukas"],
            "latt": ["poga"],
            "pols": ["guzik"],
            "mask": ["кнопка"],
        }
    }
    
    passage = make_passage(
        entry,
        include_prussian=False,
        langs=["miks", "engl", "leit", "latt", "pols", "mask"],
    )
    assert passage == "Knopf | button | mygtukas | poga | guzik | кнопка"


def test_make_passage_with_description():
    """Test passage with description appended."""
    entry = {
        "word": "test",
        "desc": "[verb]",
        "translations": {
            "engl": ["to test"],
            "miks": ["testen"],
        }
    }
    
    passage = make_passage(
        entry,
        include_prussian=True,
        include_desc=True,
        langs=["engl", "miks"],
    )
    assert passage == "test: to test | testen [verb]"


def test_make_passage_empty():
    """Test passage with no translations."""
    entry = {
        "word": "test",
        "translations": {}
    }
    
    passage = make_passage(entry, include_prussian=True)
    assert passage == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
