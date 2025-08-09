#!/usr/bin/env python
"""
update_possible_sentences.py

Usage:
    python update_possible_sentences.py <input.json> <output.json>

Creates a second file <output>.bad_ids.json containing the __id__ values
of tasks whose annotations had invalid choices that were removed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Set


###############################################################################
# 1. Sentence splitter (improved version)
###############################################################################
_ABBREVS: Set[str] = {
    "Mr", "Mrs", "Ms", "Dr", "St",
    "Prof", "Capt", "Cpt", "Lt",
    "Jr", "Sr", "Inc", "Ltd", "Co",
    # add project-specific abbreviations below if needed
}

_CLOSERS = "\"')}]”’“‘"



def _is_abbrev(word: str) -> bool:
    return word.rstrip(".").capitalize() in _ABBREVS


def split_into_sentences(text: str) -> List[str]:
    """Return a list of sentences using simple, robust heuristics."""
    sentences = []
    start = 0
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]

        if ch in ".?!":
            # Look ahead – skip blanks & closing quotes/brackets
            j = i + 1
            while j < n and text[j] == " ":
                j += 1
            k = j
            while k < n and text[k] in _CLOSERS:
                k += 1
            while k < n and text[k] == " ":
                k += 1

            # Accept split if end of text or next char is upper-case
            if k >= n or text[k].isupper():
                # Check abbrev immediately before the punctuation
                match = re.search(r"(\b\w+\b)$", text[start:i])
                if match and _is_abbrev(match.group(1)):
                    i += 1
                    continue

                # swallow immediate closing quotes, e.g. …." / …!')
                end = i + 1
                if end < n and text[end] in _CLOSERS:
                    end += 1

                sentences.append(text[start:end].strip())
                start = end
                i = start
                continue
        i += 1

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


###############################################################################
# 2. Helpers for tasks & annotations
###############################################################################
def normalise_numbering(line: str) -> str:
    """Remove leading enumerations like '1. ' or '1) '."""
    return re.sub(r"^\s*\d+[\.\)]\s*", "", line)


def build_possible_sentences(gold_text: str) -> List[dict]:
    """Create the list objects for data['possibleSentences']."""
    sentences = split_into_sentences(gold_text)
    cleaned = [normalise_numbering(s) for s in sentences if s]
    payload = [{"value": s} for s in cleaned]

    # Label Studio often keeps a generic fallback choice called “Other”
    payload.append({"value": "Other"})
    return payload


###############################################################################
# 2. Helpers for tasks & annotations   (only diff-lines are marked with ✓)
###############################################################################
def clean_annotation_results(results: List[dict], valid_choices: Set[str]) -> bool:
    """
    Remove invalid choices from a list of LS result blocks in-place.

    Returns True *only if an actually-selected choice turned out invalid* ✓
    """
    invalid_selection_found = False                         # ✓
    to_remove = []

    for r in results:
        if r.get("type") != "choices":
            continue

        # keep original ordering so we can compare faithfully
        orig = r["value"].get("choices", [])
        kept = [c for c in orig if c in valid_choices or c == "Other"]

        if kept != orig:                                   # at least one choice deleted
            invalid_selection_found = True                 # ✓ flag once per annotation
            if kept:
                r["value"]["choices"] = kept
            else:
                to_remove.append(r)

    for bad in to_remove:
        results.remove(bad)

    return invalid_selection_found                          # ✓




###############################################################################
# 3. Main transformation
###############################################################################
def transform_tasks(tasks: list) -> list:
    bad_task_ids = []

    for task in tasks:
        data = task["data"]
        gold = data.get("content", "")
        new_possible = build_possible_sentences(gold)
        data["possibleSentences"] = new_possible
        valid_choices = {item["value"] for item in new_possible}

        # --- scan annotations ---
        task_has_invalid_choice = False                     # ✓
        for ann in task.get("annotations", []):
            if clean_annotation_results(ann.get("result", []), valid_choices):
                task_has_invalid_choice = True              # ✓

        if task_has_invalid_choice:                         # ✓ only if a bad choice was present
            bad_task_ids.append(
                data.get("matching_file")                   # you preferred this field
                or data.get("__id__")
                or task.get("id")
            )

    return bad_task_ids


# Characters we want to replace -> correct equivalents
replacement_map = {
    '¡°': '“',
    '¡±': '”',
    '¡¯': '’',
    'Â¡Â¯': '’',
    'Â¡Âª': '—',
    'Â': '',   # stray byte often left after bad decoding
    '�': ''    # unknown replacement char
}

def fix_quotes(s: str) -> str:
    """Replace corrupted quote/apostrophe characters in a string."""
    if not isinstance(s, str):
        return s
    for bad, good in replacement_map.items():
        s = s.replace(bad, good)
    return s

def recurse_clean(obj):
    """Recursively walk any JSON-like structure and clean strings."""
    if isinstance(obj, dict):
        return {k: recurse_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recurse_clean(item) for item in obj]
    else:
        return fix_quotes(obj)


###############################################################################
# 4. Entry point / CLI
###############################################################################
def main() -> None:


    in_path = "../processed_data/label_studio_audio_tasks.json"
    out_path = "cleaned.json"
    in_path = Path(in_path).expanduser()
    out_path = Path(out_path).expanduser()
    bad_ids_path = out_path.with_suffix(".bad_ids.json")

    with in_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    tasks=recurse_clean(tasks)

    bad_ids = transform_tasks(tasks)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    with bad_ids_path.open("w", encoding="utf-8") as f:
        json.dump(bad_ids, f, ensure_ascii=False, indent=2)

    print(f"✔ Updated export written to {out_path}")
    print(f"✔ {len(bad_ids)} task(s) had invalid choices; list saved to {bad_ids_path}")


if __name__ == "__main__":
    main()
