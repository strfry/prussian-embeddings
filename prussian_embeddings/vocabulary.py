"""Vocabulary extraction for distillation."""

import re
from pathlib import Path
from typing import Sequence

from .passages import LANGUAGE_ORDER


def build_vocabulary(
    entries: list[dict],
    langs: Sequence[str] = LANGUAGE_ORDER,
    lowercase: bool = True,
) -> list[str]:
    """Extract and deduplicate translation words from dictionary entries.

    Tokenizes each translation using Unicode ``\\w+``, lowercases if requested,
    deduplicates and returns sorted unique tokens.

    Args:
        entries: List of dictionary entry dicts.
        langs: Language codes to include.
        lowercase: Whether to lowercase tokens.

    Returns:
        Sorted list of unique word tokens.
    """
    seen: set[str] = set()
    for entry in entries:
        translations = entry.get("translations", {})
        for lang in langs:
            for text in translations.get(lang, []):
                if not text:
                    continue
                for token in re.findall(r"\w+", text, flags=re.UNICODE):
                    if lowercase:
                        token = token.lower()
                    seen.add(token)
    return sorted(seen)


def save_vocabulary(words: Sequence[str], path: str | Path) -> None:
    """Save vocabulary to a text file (one word per line, UTF-8)."""
    Path(path).write_text("\n".join(words) + "\n", encoding="utf-8")


def load_vocabulary(path: str | Path) -> list[str]:
    """Load vocabulary from a text file (one word per line, UTF-8)."""
    return [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
