"""Dictionary-specific passage formatting for embeddings and reranking."""

from typing import Dict, List, Sequence

LANGUAGE_ORDER = ["engl", "miks", "leit", "latt", "pols", "mask"]


def has_translations(entry: Dict) -> bool:
    """Check if entry has any translations.
    
    Args:
        entry: Dictionary entry dict
        
    Returns:
        True if any translation list has at least one entry
    """
    translations = entry.get("translations", {})
    return any(
        isinstance(trans_list, list) and len(trans_list) > 0
        for trans_list in translations.values()
    )


def translations(
    entry: Dict, langs: Sequence[str] = LANGUAGE_ORDER
) -> List[str]:
    """Extract translations from entry, first one per language.
    
    Args:
        entry: Dictionary entry dict
        langs: Sequence of language codes to extract, in order
        
    Returns:
        List of translation strings, one per language in langs order
    """
    trans_dict = entry.get("translations", {})
    trans_list = []
    
    for lang_code in langs:
        if lang_code in trans_dict:
            trans = trans_dict[lang_code]
            if isinstance(trans, list) and trans:
                trans_list.append(trans[0])
    
    return trans_list


def description(entry: Dict) -> str:
    """Get description field from entry.
    
    Args:
        entry: Dictionary entry dict
        
    Returns:
        Description string (e.g., "[Advent MK]"), empty if not present
    """
    return entry.get("desc", "")


def word_type(entry: Dict) -> str:
    """Extract word type from description field.
    
    Parses the first word of the desc field and returns it lowercase.
    
    Args:
        entry: Dictionary entry dict
        
    Returns:
        Word type (lowercase), empty if not found
    """
    import re
    desc = entry.get("desc", "")
    if desc:
        match = re.match(r"^\s*(\w+)", desc)
        if match:
            return match.group(1).lower()
    return ""


def make_passage(
    entry: Dict,
    *,
    include_prussian: bool,
    include_desc: bool = False,
    langs: Sequence[str] = LANGUAGE_ORDER,
    prefix: str = "",
) -> str:
    """Format entry as passage for embedding or reranking.
    
    Two modes:
    - include_prussian=True, langs=LANGUAGE_ORDER[:4]
      → "{prefix}{word}: en | de | lt | lv"
      (for static passage embeddings)
      
    - include_prussian=False, langs=full list
      → "de | en | lt | lv | pl | ru"
      (for reranker documents, no headword)
    
    Args:
        entry: Dictionary entry dict
        include_prussian: If True, include Prussian headword
        include_desc: If True, append description to end
        langs: Sequence of language codes
        prefix: Optional prefix to prepend
        
    Returns:
        Formatted passage string, empty if no translations
    """
    trans_list = translations(entry, langs=langs)
    
    if not trans_list:
        return ""
    
    if include_prussian:
        word = entry.get("word", "")
        passage = f"{prefix}{word}: " + " | ".join(trans_list)
    else:
        passage = " | ".join(trans_list)
    
    if include_desc:
        desc = description(entry)
        if desc:
            passage += f" {desc}"
    
    return passage
