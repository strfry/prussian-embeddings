"""Tests for ref-clustered chunk building (build_chunks.py)."""

import json

import pytest

from prussian_embeddings.build_chunks import (
    build_chunks,
    build_clusters,
    entry_pos,
    load_links,
    pos_from_desc,
    pos_from_tags,
    read_chunks,
    write_chunks,
)


# ── POS from FST tags ──

def test_pos_from_tags_category():
    assert pos_from_tags(["N+Sg+Akk+Masc"]) == "noun"
    assert pos_from_tags(["V+Ind+Pres+P1+Pl"]) == "verb"
    assert pos_from_tags(["Adj+Sg+Nom"]) == "adjective"
    assert pos_from_tags(["Adv"]) == "adverb"
    assert pos_from_tags(["Num+Sg+Nom"]) == "numeral"


def test_pos_from_tags_participle_wins():
    # A participle is a verb analysis with +Part; report it distinctly.
    assert pos_from_tags(["V+Part+Pres+Sg+Nom+Masc"]) == "participle"


def test_pos_from_tags_real_categories():
    # Tag names as emitted by the prussian-fst analyzer.
    assert pos_from_tags(["PropN+Sg+Nom+Masc"]) == "noun"
    assert pos_from_tags(["Prp"]) == "preposition"
    assert pos_from_tags(["IJ"]) == "interjection"
    assert pos_from_tags(["Cnj"]) == "conjunction"


def test_pos_from_tags_priority_order_independent():
    # A word with both noun and verb readings → verb (higher priority),
    # regardless of order.
    assert pos_from_tags(["N+Sg+Nom", "V+Inf"]) == "verb"
    assert pos_from_tags(["V+Inf", "N+Sg+Nom"]) == "verb"
    # Adjective + noun → noun.
    assert pos_from_tags(["Adj+Sg+Nom", "N+Sg+Nom"]) == "noun"


def test_pos_from_tags_none():
    assert pos_from_tags([]) is None
    assert pos_from_tags(["Zzz+Foo"]) is None


# ── POS from desc marker (fallback) ──

def test_pos_from_desc_marker():
    assert pos_from_desc("aj [Adjektiv]") == "adjective"
    assert pos_from_desc("av") == "adverb"
    assert pos_from_desc("crd 5") == "numeral"
    assert pos_from_desc("[aj drv]") == "adjective"  # leading bracket stripped


def test_pos_from_desc_none():
    assert pos_from_desc("") is None
    assert pos_from_desc("Advent MK") is None


# ── entry_pos priority ──

def test_entry_pos_prefers_own_fst_tags():
    entry = {"word": "arktisks", "desc": "aj"}
    tags_by_word = {"arktisks": ["Adj+Sg+Nom"]}
    # Own FST tags win even when a link/desc would say otherwise.
    assert entry_pos(entry, {}, tags_by_word) == ("adjective", "fst")


def test_entry_pos_falls_back_to_link_then_desc():
    entry = {"word": "madlimai", "desc": "ps"}
    links = {"madlimai": [{"lemma": "madlītun", "tags": ["V+Ind+Pres+P1+Pl"]}]}
    assert entry_pos(entry, links, None) == ("verb", "fst")
    # No link, no tags → desc marker.
    assert entry_pos({"word": "x", "desc": "av"}, {}, None) == ("adverb", "desc")
    # Nothing at all.
    assert entry_pos({"word": "x", "desc": ""}, {}, None) == (None, None)


# ── clustering ──

def _entries():
    return [
        {"word": "madlītun", "desc": "", "translations": {"engl": ["to pray"]}},
        {"word": "madlimai", "desc": "ps", "translations": {}},
        {"word": "arktisks", "desc": "aj", "translations": {"engl": ["arctic"]}},
    ]


def _members_by_lemma(clusters):
    return {c["lemma"]: [e["word"] for e in c["members"]] for c in clusters.values()}


def test_build_clusters_groups_form_under_lemma():
    entries = _entries()
    links = {"madlimai": [{"lemma": "madlītun", "tags": ["V+Ind+Pres+P1+Pl"]}]}
    by_lemma = _members_by_lemma(build_clusters(entries, links))
    # madlītun cluster gets both the base entry and the resolved form.
    assert by_lemma["madlītun"] == ["madlītun", "madlimai"]
    # arktisks is its own singleton.
    assert by_lemma["arktisks"] == ["arktisks"]


def test_build_clusters_ignores_self_link():
    entries = [{"word": "leītun", "translations": {"engl": ["to pour"]}}]
    # A link whose lemma equals the word must not create a phantom cluster.
    links = {"leītun": [{"lemma": "leītun", "tags": ["V+Inf"]}]}
    clusters = build_clusters(entries, links)
    assert _members_by_lemma(clusters) == {"leītun": ["leītun"]}


def test_build_clusters_homographs_not_merged():
    # Two same-spelled standalone entries must stay in separate chunks.
    entries = [
        {"word": "grēks", "translations": {"engl": ["Greek man"]}},
        {"word": "grēks", "translations": {"engl": ["sin"]}},
    ]
    clusters = build_clusters(entries, {})
    assert len(clusters) == 2


def test_build_clusters_links_keyed_by_entry_id():
    # Homographs share the word, not the id: only be-2's link groups it.
    entries = [
        {"id": "be-1", "word": "be", "translations": {"en": ["without"]}},
        {"id": "be-2", "word": "be", "translations": {"en": ["and"]}},
        {"id": "bhe", "word": "bhe", "translations": {"en": ["and"]}},
    ]
    links = {"be-2": [{"lemma": "bhe", "tags": ["Cnj"]}]}
    by_lemma = {c["lemma"]: [e["id"] for e in c["members"]]
                for c in build_clusters(entries, links).values()}
    # The base entry anchors its cluster, first in line.
    assert by_lemma == {"be": ["be-1"], "bhe": ["bhe", "be-2"]}


def test_build_clusters_form_joins_each_target():
    # A form linked to two same-POS homographs appears in both chunks; the
    # homographs themselves stay apart.
    entries = [
        {"id": "bilītun-1", "word": "bilītun", "translations": {"en": ["speak"]}},
        {"id": "bilītun-2", "word": "bilītun", "translations": {"en": ["mean"]}},
        {"id": "bilāi", "word": "bilāi", "translations": {}},
    ]
    links = {"bilāi": [{"lemma": "bilītun",
                        "target_ids": ["bilītun-1", "bilītun-2"]}]}
    members = [[e["id"] for e in c["members"]]
               for c in build_clusters(entries, links).values()]
    assert members == [["bilītun-1", "bilāi"], ["bilītun-2", "bilāi"]]


def test_build_clusters_follows_chain_to_base():
    entries = [
        {"id": "walns", "word": "walns"},
        {"id": "walnan", "word": "walnan"},
        {"id": "walnanin", "word": "walnanin"},
    ]
    links = {"walnan": [{"lemma": "walns", "target_ids": ["walns"]}],
             "walnanin": [{"lemma": "walnan", "target_ids": ["walnan"]}]}
    members = [[e["id"] for e in c["members"]]
               for c in build_clusters(entries, links).values()]
    assert members == [["walns", "walnan", "walnanin"]]


# ── full chunk build ──

def test_build_chunks_text_and_pos():
    entries = _entries()
    links = {"madlimai": [{"lemma": "madlītun", "tags": ["V+Ind+Pres+P1+Pl"]}]}
    tags = {"madlītun": ["V+Inf"], "arktisks": ["Adj+Sg+Nom"]}
    chunks = build_chunks(entries, links, tags, langs=["engl"])
    by_lemma = {c["lemma"]: c for c in chunks}

    verb = by_lemma["madlītun"]
    assert verb["members"] == ["madlītun", "madlimai"]
    assert verb["pos"] == "verb"
    assert verb["pos_source"] == "fst"
    # Base lemma line has translations; the formless inflection still appears.
    assert "madlītun (verb): to pray" in verb["text"]
    assert "madlimai (verb)" in verb["text"]

    adj = by_lemma["arktisks"]
    assert adj["pos"] == "adjective"
    assert adj["text"] == "arktisks (adjective): arctic"


def test_build_chunks_member_ids_and_nvh_pos():
    entries = [
        {"id": "be-1", "word": "be", "pos": "prep",
         "translations": {"en": ["without"], "de": ["ohne"]}},
        {"id": "be-2", "word": "be", "pos": "cconj",
         "translations": {"en": ["and"], "de": ["und"]}},
    ]
    # NVH pos wins over FST tags of the word.
    chunks = build_chunks(entries, {}, {"be": ["Prp"]})
    assert [c["member_ids"] for c in chunks] == [["be-1"], ["be-2"]]
    assert [(c["pos"], c["pos_source"]) for c in chunks] == [
        ("preposition", "nvh"), ("conjunction", "nvh")]
    assert chunks[1]["text"] == "be (conjunction): and | und"


def test_build_chunks_form_line_keeps_only_grammar():
    entries = [
        {"id": "wīrs", "word": "wīrs", "translations": {"en": ["man"]}},
        {"id": "wīrans", "word": "wīrans", "desc": "↑ Wīrs acc sg m [Wīrans 93(12)]"},
        {"id": "klāntamai", "word": "klāntamai",
         "desc": "↑ Klantītun, ↑ Wīrs ps 1 sg m [klantemmai 29]"},
    ]
    links = {"wīrans": [{"ref": "Wīrs", "lemma": "wīrs", "target_ids": ["wīrs"]}],
             "klāntamai": [{"ref": "Wīrs", "lemma": "wīrs", "target_ids": ["wīrs"]}]}
    [chunk] = build_chunks(entries, links, None, langs=["en"])
    assert chunk["text"].splitlines()[1:] == ["wīrans acc sg m", "klāntamai ps 1 sg m"]


def test_build_chunks_passthrough_without_pos():
    # Unresolved / unanalyzable entry is passed through as a singleton, no POS.
    entries = [{"word": "parēitwei", "desc": "", "translations": {}}]
    chunks = build_chunks(entries, {}, None)
    assert len(chunks) == 1
    assert chunks[0]["pos"] is None
    assert chunks[0]["pos_source"] is None
    assert chunks[0]["text"] == "parēitwei"


# ── IO round-trip ──

def test_write_read_chunks_roundtrip(tmp_path):
    chunks = build_chunks(_entries(), {}, None, langs=["engl"])
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    assert read_chunks(path) == chunks


def test_load_links_drops_unresolved(tmp_path):
    records = [
        {"orig_lemma": "madlimai", "lemma": "madlītun", "tags": ["V+Ind"]},
        {"orig_lemma": "foo", "status": "gap"},  # unresolved, no lemma
        {"orig_lemma": "bar", "status": "ambiguous", "candidates": ["a", "b"]},
    ]
    path = tmp_path / "links.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    links = load_links(path)
    assert set(links) == {"madlimai"}


def test_load_links_lemmas_schema(tmp_path):
    # The linker emits a `lemmas` list; the first is used as the grouping lemma.
    records = [
        {"orig_lemma": "belarussiskan", "ref": "belarussisks",
         "lemmas": ["belarussisks"], "tags": ["Adj+Sg+Nom+Masc"], "method": "exact"},
        {"orig_lemma": "empty", "lemmas": [], "method": "gap"},  # dropped
    ]
    path = tmp_path / "links.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    links = load_links(path)
    assert set(links) == {"belarussiskan"}
    assert links["belarussiskan"][0]["lemma"] == "belarussisks"


def test_chunks_embed_into_store():
    """The generate --chunks data flow: chunk text → store with chunk records."""
    import numpy as np

    from prussian_embeddings.store import EmbeddingStore

    class FakeEmbedder:
        dim = 4

        def get_embeddings(self, texts):
            return np.array(
                [[float(len(t)), 1.0, 2.0, 3.0] for t in texts], dtype=np.float32
            )

    chunks = build_chunks(_entries(), {}, None, langs=["engl"])
    texts = ["passage: " + c["text"] for c in chunks]
    store = EmbeddingStore.build(FakeEmbedder(), texts=texts, records=chunks)
    store.meta = {"strategy": "chunks", "num_entries": len(chunks)}

    assert store.embeddings.shape == (len(chunks), 4)
    assert store.records[0]["lemma"] == chunks[0]["lemma"]
    assert store.meta["strategy"] == "chunks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
