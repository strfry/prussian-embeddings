"""Build ref-clustered embedding chunks from the dictionary + FST linker output.

Instead of embedding each dictionary entry in isolation, entries that share a
base lemma are grouped into one *chunk* and embedded together. The grouping is
driven by the FST linker (``prussian-fst``, ``prussian_fst.linker``), which
resolves each entry's ``desc`` reference to an FST lemma. Forms and derivatives
that resolve to the same lemma land in the same chunk, so a translation-less
inflected form becomes findable through the co-embedded base lemma.

Two FST-produced inputs (the resolution work is delegated to the linker, not
re-implemented here):

- ``links.json`` — the linker's resolved links. One record per resolved
  ``desc`` reference::

      {"orig_lemma": <entry word>, "ref": <ref form>, "lemmas": [<FST lemma>, ...],
       "tags": ["V+Ind+Pres+P1+Pl", ...], "method": "macron", ...}

  ``lemmas`` is a list (usually one; several when the linker resolves a
  cluster) — the first is used as the grouping lemma. The older singular
  ``lemma`` key is still accepted. ``links_unresolved.json`` (status ``gap`` /
  ``ambiguous``) is ignored by design.

- ``tags.json`` (optional) — headword → FST analyses, produced by
  ``prussian_fst.api.tags``: ``{<word>: ["Adj+Sg+Nom", ...]}``. When present it
  is the authoritative part-of-speech source for that word.

Part-of-speech per entry is resolved in priority order: the entry's own FST
tags (``tags.json``) → the tags of its resolved link → a ``desc`` marker
fallback → none. Unresolved / unanalyzable entries simply carry no POS label
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

from .passages import translations, word_type

DEFAULT_LANGS = ["engl", "miks", "leit", "latt"]

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


def entry_pos(
    entry: Dict,
    links_by_orig: Dict[str, List[Dict]],
    tags_by_word: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (pos, source) for one entry.

    Priority: the word's own FST analysis (``tags.json``) → the FST tags of its
    resolved link → a ``desc`` marker → ``(None, None)``.
    """
    word = entry.get("word", "")
    if tags_by_word:
        pos = pos_from_tags(tags_by_word.get(word, []))
        if pos:
            return pos, "fst"
    for rec in links_by_orig.get(word, []):
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
    """Load the linker's ``links.json`` into ``{orig_lemma: [resolved record]}``.

    Each kept record is normalized to carry a singular ``lemma`` (first of
    ``lemmas``). Unresolved records (no lemma) are dropped — ignored by design.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    by_orig: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        lemma = _rec_lemma(rec)
        if lemma and rec.get("orig_lemma"):
            rec = {**rec, "lemma": lemma}
            by_orig[rec["orig_lemma"]].append(rec)
    return by_orig


def load_tags(path: str | Path) -> Dict[str, List[str]]:
    """Load an optional ``tags.json`` (``{word: [tag_string, ...]}``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_clusters(
    entries: List[Dict],
    links_by_orig: Dict[str, List[Dict]],
) -> "OrderedDict[str, Dict]":
    """Group entries into ref-clusters keyed by their resolved base lemma.

    Only genuine link relationships group entries:

    - An entry with a resolved link to lemma L (L ≠ its own word) joins the
      cluster of L.
    - An entry whose own word is the target of some other entry's link anchors
      that lemma's cluster.
    - Every other entry is a singleton — kept separate even from same-spelled
      homographs (they are not merged by bare spelling).

    Insertion order is preserved so output is deterministic. The internal
    ``OrderedDict`` key is unique per cluster; ``cluster["lemma"]`` is the
    display headword.
    """
    # Lemmas that some entry resolves to (normalized) → forms will attach here.
    target_lemmas = {
        rec["lemma"].casefold()
        for recs in links_by_orig.values()
        for rec in recs
    }

    clusters: "OrderedDict[str, Dict]" = OrderedDict()

    def cluster_for(key: str, lemma: str) -> Dict:
        if key not in clusters:
            clusters[key] = {"lemma": lemma, "members": []}
        return clusters[key]

    for idx, entry in enumerate(entries):
        word = entry["word"]
        wfold = word.casefold()

        # A resolved link to a different lemma → join that lemma's cluster.
        target = None
        for rec in links_by_orig.get(word, []):
            if rec["lemma"].casefold() != wfold:
                target = rec["lemma"]
                break

        if target is not None:
            cluster_for(f"lemma:{target.casefold()}", target)["members"].append(entry)
        elif wfold in target_lemmas:
            # This entry is a base lemma other forms point at — anchor it.
            cluster_for(f"lemma:{wfold}", word)["members"].append(entry)
        else:
            # Standalone entry: unique key so homographs never merge.
            cluster_for(f"solo:{idx}:{word}", word)["members"].append(entry)

    return clusters


def _entry_line(
    entry: Dict,
    pos: Optional[str],
    langs: Sequence[str],
) -> str:
    """One chunk line for an entry: ``word (pos): tr | tr`` or ``word (pos) desc``.

    Translation-less forms still get a line (word + optional desc) so the base
    lemma's chunk exposes the inflected/derived form to the embedder.
    """
    word = entry.get("word", "")
    head = f"{word} ({pos})" if pos else word
    trans = translations(entry, langs=langs)
    if trans:
        return f"{head}: " + " | ".join(trans)
    desc = (entry.get("desc") or "").strip()
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
         "pos": <pos or None>, "pos_source": <"fst"|"desc"|None>,
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
            lines.append(_entry_line(entry, pos, langs))
            # Prefer the anchor (word == cluster lemma) for the chunk POS.
            is_anchor = entry["word"].casefold() == lemma.casefold()
            if pos and (chunk_pos is None or is_anchor):
                chunk_pos, chunk_src = pos, src
        chunks.append(
            {
                "lemma": lemma,
                "members": [e["word"] for e in members],
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


def _load_entries(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise SystemExit("ERROR: expected list or dict in dictionary JSON")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ref-clustered embedding chunks (dictionary + FST links)"
    )
    parser.add_argument(
        "--dictionary",
        default="../corpus/parsed/twanksta_entries.json",
        help="Path to dictionary JSON (twanksta_entries.json)",
    )
    parser.add_argument(
        "--links",
        required=True,
        help="Path to the FST linker's links.json",
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

    entries = _load_entries(dict_path)
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
