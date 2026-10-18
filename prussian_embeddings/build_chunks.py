"""Build ref-clustered embedding chunks from the dictionary + FST linker output.

The dictionary is the lean Twanksta NVH (``../corpus/parsed/twanksta_dmlex.nvh``),
read via ``prussian_fst.nvh``. Every entry carries its NVH ``id`` (``be-1``,
``be-2`` for homographs); chunks list them as ``member_ids`` next to the
``members`` words, so a consumer can show exactly the embedded entries.

Instead of embedding each dictionary entry in isolation, entries that share a
base lemma are grouped into one *chunk* and embedded together. The grouping is
driven by the linker (``prussian-fst``, ``prussian_fst.linker``), which
resolves each inflected-form entry's ``↑`` reference (``wīrans ↑ Wīrs acc sg
m``) to its base entry. A form lands in its base entry's chunk, so a
translation-less inflected form becomes findable through the co-embedded base
lemma. Derivatives and compounds stay separate chunks.

Two FST-produced inputs (the resolution work is delegated to the linker, not
re-implemented here):

- ``links.json`` — the linker's resolved links. One record per resolved
  ``↑`` reference::

      {"orig_lemma": <entry word>, "orig_id": <entry id>, "ref": <ref form>,
       "target_ids": [<base entry id>, ...], "lemmas": [<base word>],
       "tags": ["N+Msc+Sg+Acc", ...], "method": "headword", ...}

  ``target_ids`` names the base entries (several for homographs of matching
  part of speech). Records without it fall back to all entries of ``lemmas[0]``
  (or the older singular ``lemma``). ``links_unresolved.json`` is ignored by
  design.

- ``tags.json`` (optional) — headword → FST analyses, produced by
  ``prussian_fst.api.tags``: ``{<word>: ["Adj+Sg+Nom", ...]}``. When present it
  is the authoritative part-of-speech source for that word.

Part-of-speech per entry is resolved in priority order: the entry's NVH
``pos`` → its own FST tags (``tags.json``) → the tags of its resolved link → a
``desc`` marker fallback → none. Unresolved / unanalyzable entries simply carry no POS label
and are passed through as singleton chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from prussian_fst.nvh import DEFAULT_LEAN_NVH, read_entries

from .passages import translations, word_type

DEFAULT_LANGS = ["en", "de", "lt", "lv"]

# NVH ``pos`` value → written-out part of speech.
NVH_POS = {
    "noun": "noun",
    "verb": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "num": "numeral",
    "pron": "pronoun",
    "prep": "preposition",
    "postp": "postposition",
    "cconj": "conjunction",
    "sconj": "conjunction",
    "intj": "interjection",
    "part": "particle",
}

# FST top-level category tag → written-out part of speech (tag names as emitted
# by the prussian-fst analyzer).
FST_POS = {
    "V": "verb",
    "N": "noun",
    "PropN": "noun",
    "Adj": "adjective",
    "Num": "numeral",
    "Adv": "adverb",
    "Pron": "pronoun",
    "Prp": "preposition",
    "Psp": "postposition",
    "Cnj": "conjunction",
    "SCnj": "conjunction",
    "IJ": "interjection",
    "Pcl": "particle",
}

# When a word has several readings, the most specific / most informative POS
# wins deterministically (order-independent).
POS_PRIORITY = [
    "participle",
    "verb",
    "noun",
    "adjective",
    "numeral",
    "adverb",
    "pronoun",
    "preposition",
    "postposition",
    "conjunction",
    "interjection",
    "particle",
]

# Twanksta desc first-token marker → part of speech (fallback only).
DESC_MARKER_POS = {
    "aj": "adjective",
    "av": "adverb",
    "adv": "adverb",
    "pc": "participle",
    "crd": "numeral",
    "ord": "numeral",
    "num": "numeral",
    "prep": "preposition",
    "prp": "preposition",
    "cnj": "conjunction",
    "conj": "conjunction",
    "encl": "particle",
    "pn": "pronoun",
    "pnl": "pronoun",
    "pron": "pronoun",
    "ij": "interjection",
    "n": "noun",
    "no": "noun",
    "subst": "noun",
}


def pos_from_tags(tags: Sequence[str]) -> Optional[str]:
    """Map FST tag strings (e.g. ``"V+Part+Pres+Sg+Nom"``) to a single POS.

    All readings are considered; the highest-priority POS present wins
    (``participle`` on any ``+Part`` reading, then verb, noun, …), so the result
    does not depend on reading order.
    """
    found = set()
    for tag in tags:
        comps = tag.split("+")
        if "Part" in comps:
            found.add("participle")
        cat = comps[0]
        if cat in FST_POS:
            found.add(FST_POS[cat])
    for pos in POS_PRIORITY:
        if pos in found:
            return pos
    return None


def pos_from_desc(desc: str) -> Optional[str]:
    """Fallback POS from the first token of a twanksta ``desc`` field."""
    marker = word_type({"desc": desc.lstrip("[")}) if desc else ""
    return DESC_MARKER_POS.get(marker)


def entry_key(entry: Dict) -> str:
    """Identity of an entry: its NVH ``id``, else the bare word."""
    return entry.get("id") or entry.get("word", "")


def entry_pos(
    entry: Dict,
    links_by_orig: Dict[str, List[Dict]],
    tags_by_word: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (pos, source) for one entry.

    Priority: the NVH ``pos`` → the word's own FST analysis (``tags.json``) →
    the FST tags of its resolved link → a ``desc`` marker → ``(None, None)``.
    """
    word = entry.get("word", "")
    pos = NVH_POS.get(entry.get("pos", ""))
    if pos:
        return pos, "nvh"
    if tags_by_word:
        pos = pos_from_tags(tags_by_word.get(word, []))
        if pos:
            return pos, "fst"
    for rec in links_by_orig.get(entry_key(entry), []):
        pos = pos_from_tags(rec.get("tags", []))
        if pos:
            return pos, "fst"
    pos = pos_from_desc(entry.get("desc", ""))
    if pos:
        return pos, "desc"
    return None, None


def _rec_lemma(rec: Dict) -> Optional[str]:
    """Primary resolved lemma of a link record (``lemmas[0]`` or legacy ``lemma``)."""
    lemmas = rec.get("lemmas")
    if lemmas:
        return lemmas[0]
    return rec.get("lemma")


def load_links(path: str | Path) -> Dict[str, List[Dict]]:
    """Load the linker's ``links.json`` into ``{orig_id: [resolved record]}``.

    Records are keyed by the entry id (``orig_id``), falling back to
    ``orig_lemma`` for links without one. Each kept record is normalized to
    carry a singular ``lemma`` (first of ``lemmas``). Unresolved records (no
    lemma) are dropped — ignored by design.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    by_orig: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        lemma = _rec_lemma(rec)
        key = rec.get("orig_id") or rec.get("orig_lemma")
        if lemma and key:
            by_orig[key].append({**rec, "lemma": lemma})
    return by_orig


def load_tags(path: str | Path) -> Dict[str, List[str]]:
    """Load an optional ``tags.json`` (``{word: [tag_string, ...]}``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _target_keys(rec: Dict, keys_by_word: Dict[str, List[str]]) -> List[str]:
    """Entry keys a link points at: its ``target_ids``, else all entries of its lemma."""
    if rec.get("target_ids"):
        return list(rec["target_ids"])
    return keys_by_word.get(rec["lemma"].casefold(), [])


def build_clusters(
    entries: List[Dict],
    links_by_orig: Dict[str, List[Dict]],
) -> "OrderedDict[str, Dict]":
    """Group entries into ref-clusters, one per linked base entry.

    Only genuine link relationships group entries:

    - An entry with resolved links joins the cluster of each target entry
      (``target_ids``; several when the linker kept more than one homograph,
      so the form then appears in each of their chunks).
    - Chains (form → form → base) are followed to the base entry.
    - A target entry anchors its own cluster, first in line.
    - Every other entry is a singleton — kept separate even from same-spelled
      homographs (they are not merged by bare spelling).

    Insertion order follows the entries, so output is deterministic. The
    internal ``OrderedDict`` key is unique per cluster; ``cluster["lemma"]`` is
    the display headword.
    """
    by_key = {entry_key(e): e for e in entries}
    keys_by_word: Dict[str, List[str]] = defaultdict(list)
    for e in entries:
        keys_by_word[e["word"].casefold()].append(entry_key(e))

    targets_of: Dict[str, List[str]] = {}
    for e in entries:
        key = entry_key(e)
        targets = [t for rec in links_by_orig.get(key, [])
                   for t in _target_keys(rec, keys_by_word)
                   if t != key and t in by_key]
        if targets:
            targets_of[key] = list(dict.fromkeys(targets))

    def base_of(key: str, seen: frozenset = frozenset()) -> List[str]:
        # Follow chains (form → form → base) to entries that link nowhere.
        if key not in targets_of or key in seen:
            return [key]
        return [b for t in targets_of[key] for b in base_of(t, seen | {key})]

    targets_of = {k: list(dict.fromkeys(b for t in ts for b in base_of(t, frozenset({k}))
                                        if b != k))
                  for k, ts in targets_of.items()}
    targets_of = {k: ts for k, ts in targets_of.items() if ts}
    anchors = {t for ts in targets_of.values() for t in ts}

    clusters: "OrderedDict[str, Dict]" = OrderedDict()

    def cluster_for(key: str) -> Dict:
        if key not in clusters:
            clusters[key] = {"lemma": by_key[key]["word"], "members": []}
        return clusters[key]

    for idx, entry in enumerate(entries):
        key = entry_key(entry)
        if key in targets_of:
            for target in targets_of[key]:
                cluster_for(target)["members"].append(entry)
        elif key in anchors:
            cluster_for(key)["members"].insert(0, entry)
        else:
            # Standalone entry: unique key so homographs never merge.
            clusters[f"solo:{idx}:{key}"] = {"lemma": entry["word"],
                                             "members": [entry]}

    return clusters


def _form_grammar(desc: str, refs: Sequence[str]) -> str:
    """A linked form's desc reduced to its grammar: ``↑ Wīrs acc sg m
    [Wīrans 93]`` → ``acc sg m``. The target is the chunk's anchor anyway and
    the ``[...]`` attestations only spend the embedder's token budget."""
    desc = re.sub(r"\[[^\]]*\]", " ", desc)
    for ref in refs:
        desc = desc.replace(f"↑ {ref}", " ")
    desc = re.sub(r"↑\s*\S+", " ", desc)
    return re.sub(r"\s+", " ", desc).strip(" ,")


def _entry_line(
    entry: Dict,
    pos: Optional[str],
    langs: Sequence[str],
    refs: Sequence[str] = (),
) -> str:
    """One chunk line for an entry: ``word (pos): tr | tr`` or ``word (pos) desc``.

    Translation-less forms still get a line (word + optional desc) so the base
    lemma's chunk exposes the inflected/derived form to the embedder. A linked
    form (``refs``) shows only the grammar part of its desc.
    """
    word = entry.get("word", "")
    head = f"{word} ({pos})" if pos else word
    trans = translations(entry, langs=langs)
    if trans:
        return f"{head}: " + " | ".join(trans)
    desc = (entry.get("desc") or "").strip()
    if refs:
        desc = _form_grammar(desc, refs)
    return f"{head} {desc}".strip() if desc else head


def build_chunks(
    entries: List[Dict],
    links_by_orig: Dict[str, List[Dict]],
    tags_by_word: Optional[Dict[str, List[str]]] = None,
    *,
    langs: Sequence[str] = DEFAULT_LANGS,
) -> List[Dict]:
    """Build chunk records ready to embed.

    Each chunk::

        {"lemma": <cluster key>, "members": [word, ...],
         "member_ids": [entry id, ...],
         "pos": <pos or None>, "pos_source": <"nvh"|"fst"|"desc"|None>,
         "text": "<line>\\n<line>..."}

    The chunk ``pos`` is taken from the member matching the cluster lemma (the
    anchor), else the first member with a POS.
    """
    clusters = build_clusters(entries, links_by_orig)
    chunks: List[Dict] = []
    for cluster in clusters.values():
        members = cluster["members"]
        lemma = cluster["lemma"]
        lines: List[str] = []
        chunk_pos: Optional[str] = None
        chunk_src: Optional[str] = None
        for entry in members:
            pos, src = entry_pos(entry, links_by_orig, tags_by_word)
            refs = [rec.get("ref", "")
                    for rec in links_by_orig.get(entry_key(entry), [])]
            lines.append(_entry_line(entry, pos, langs, [r for r in refs if r]))
            # Prefer the anchor (word == cluster lemma) for the chunk POS.
            is_anchor = entry["word"].casefold() == lemma.casefold()
            if pos and (chunk_pos is None or is_anchor):
                chunk_pos, chunk_src = pos, src
        chunks.append(
            {
                "lemma": lemma,
                "members": [e["word"] for e in members],
                "member_ids": [entry_key(e) for e in members],
                "pos": chunk_pos,
                "pos_source": chunk_src,
                "text": "\n".join(lines),
            }
        )
    return chunks


def write_chunks(chunks: List[Dict], path: str | Path) -> None:
    """Write chunks as JSON Lines."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_chunks(path: str | Path) -> List[Dict]:
    """Read a JSON Lines chunk file."""
    chunks: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ref-clustered embedding chunks (dictionary + FST links)"
    )
    parser.add_argument(
        "--dictionary",
        default=str(DEFAULT_LEAN_NVH),
        help="Path to the lean NVH (default: ../corpus/parsed/twanksta_dmlex.nvh)",
    )
    parser.add_argument(
        "--links",
        default="../fst/build/links.json",
        help="Path to the FST linker's links.json (default: ../fst/build/links.json)",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Optional tags.json (word -> FST tag strings) for headword POS",
    )
    parser.add_argument(
        "--out",
        default="data/embedding_chunks.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated language codes for translations",
    )
    args = parser.parse_args()

    dict_path = Path(args.dictionary)
    if not dict_path.exists():
        print(f"ERROR: dictionary not found at {dict_path}", file=sys.stderr)
        sys.exit(1)

    entries = read_entries(dict_path)
    links_by_orig = load_links(args.links)
    tags_by_word = load_tags(args.tags) if args.tags else None
    langs = [c.strip() for c in args.langs.split(",") if c.strip()]

    chunks = build_chunks(entries, links_by_orig, tags_by_word, langs=langs)
    write_chunks(chunks, args.out)

    n_multi = sum(1 for c in chunks if len(c["members"]) > 1)
    n_pos = sum(1 for c in chunks if c["pos"])
    grouped = sum(len(c["members"]) for c in chunks if len(c["members"]) > 1)
    print(f"entries:        {len(entries)}")
    print(f"chunks:         {len(chunks)}")
    print(f"  multi-member: {n_multi} (covering {grouped} entries)")
    print(f"  with POS:     {n_pos} ({100 * n_pos / max(1, len(chunks)):.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
