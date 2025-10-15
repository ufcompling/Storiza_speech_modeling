import json
import math
from typing import List, Dict, Optional, Tuple, Union, Set
from pathlib import Path

import pandas as pd


# Target MFA (IPA) phone inventory from the user
MFA_IPA_INVENTORY = [
    "aj","aw","b","bʲ","c","cʰ","cʷ","d","dʒ","dʲ","d̪","ej","f","fʲ","h","i","iː","j",
    "k","kʰ","kʷ","l","m","mʲ","m̩","n","n̩","ow","p","pʰ","pʲ","pʷ","s","t","tʃ","tʰ",
    "tʲ","tʷ","t̪","v","vʲ","w","z","æ","ç","ð","ŋ","ɐ","ɑ","ɑː","ɒ","ɒː","ɔj","ə","ɚ",
    "ɛ","ɝ","ɟ","ɟʷ","ɡ","ɡʷ","ɪ","ɫ","ɫ̩","ɱ","ɲ","ɹ","ɾ","ɾʲ","ɾ̃","ʃ","ʉ","ʉː","ʊ",
    "ʎ","ʒ","ʔ","θ"
]
MFA_IPA_SET: Set[str] = set(MFA_IPA_INVENTORY)

DIPHTHONG_MAP = {
    "eɪ": "ej",
    "aɪ": "aj",
    "oʊ": "ow",
    "ɔɪ": "ɔj",

}


def normalize_diphthongs(s: str) -> str:
    # Apply all replacements; longer keys first to avoid partial overlaps
    items = sorted(DIPHTHONG_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
    for k, v in items:
        s = s.replace(k, v)
    return s

def greedy_tokenize_to_inventory(ipa_str: str, inventory: List[str]) -> List[str]:
    """
    Greedily scan ipa_str (no spaces) into phones contained in inventory.
    Unknown spans are collapsed into 'spn' tokens.
    """
    phones_sorted = sorted(inventory, key=len, reverse=True)
    i = 0
    tokens: List[str] = []
    unknown_buf = ""
    while i < len(ipa_str):
        matched = None
        for ph in phones_sorted:
            L = len(ph)
            if ipa_str[i:i+L] == ph:
                matched = ph
                break
        if matched:
            # flush unknown buffer if any
            if unknown_buf.strip():
                #tokens.append("spn")
                unknown_buf = ""
            tokens.append(matched)
            i += len(matched)
        else:
            unknown_buf += ipa_str[i]
            i += 1
    # if unknown_buf.strip():
    #     tokens.append("spn")
    return tokens

def split_raw_candidates(raw_ipa: str) -> List[str]:
    """Split raw IPA string by common separators into candidate strings."""
    if not isinstance(raw_ipa, str) or not raw_ipa.strip():
        return []
    # Normalize whitespace, then split by separators
    s = raw_ipa.strip()
    if ";" in s:
        parts = s.split(";")
    elif "\n" in s:
        parts = s.splitlines()
    else:
        parts = [s]
    # Trim each
    parts = [" ".join(p.strip().split()) for p in parts if p and p.strip()]
    return parts

def ipa_candidates_from_raw(
    raw_ipa: str,
    inventory: List[str] = MFA_IPA_INVENTORY,
    treat_stutter_as_cutoff: bool = True,
) -> List[str]:
    """
    Turn a raw IPA field into a list of MFA-ready candidate phone sequences.
    - Splits by '||' / ';' / newlines
    - Normalizes diphthongs to match the inventory (e.g., eɪ->ej)
    - Greedily tokenizes to inventory (unknown chunks become 'spn')
    - If '.' or '|' are present (stutter/cutoff), generate **prefix variants**
      up to the marker to mimic cutoff pronunciations, plus a full version
      with markers removed.
    Returns: list of space-separated phone strings.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for part in split_raw_candidates(raw_ipa):
        # Remove spaces for tokenization stage; we'll re-space later
        # But keep markers '.' and '|' if we want cutoff modeling
        # We'll create a marker-stripped version and (optionally) prefix variants.
        # First normalize diphthongs on a no-space copy.
        no_space = "".join(part.split())
        no_space = normalize_diphthongs(no_space)

        # Identify stutter/cutoff markers
        marker_positions = []
        if treat_stutter_as_cutoff:
            for idx, ch in enumerate(no_space):
                if ch in {".", "|"}:
                    marker_positions.append(idx)

        # Build a base string with markers removed for the "full" candidate
        no_space_no_marker=no_space.replace(".","ʔ").replace("|","ʔ")
        base_no_markers = "".join(ch for ch in no_space_no_marker if ch not in {".", "|"})
        base_tokens = greedy_tokenize_to_inventory(base_no_markers, inventory)
        if base_tokens:
            cand = " ".join(base_tokens)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)

        # Generate cutoff prefix variants: for each marker, take everything BEFORE marker
        # and tokenize it; this mimics how MFA's cutoff model produces subsequences.
        # if treat_stutter_as_cutoff and marker_positions:
        #     for mpos in marker_positions:
        #         prefix = no_space[:mpos]  # everything before marker
        #         if not prefix:
        #             continue
        #         prefix_tokens = greedy_tokenize_to_inventory(prefix, inventory)
        #         if not prefix_tokens:
        #             continue
        #         cand = " ".join(prefix_tokens)
        #         if cand and cand not in seen:
        #             seen.add(cand)
        #             out.append(cand)

    return out




def generate_mfa_items(
        csv_path: Union[str, Path],
        directory_root: Union[str, Path],
        item_range: Optional[Tuple[int, int]] = None,
        strip_filename: bool = True,
) -> List[Dict]:
    """
    Build MFA input items from a CSV in the format you provided.

    Args:
        csv_path: path to CSV with columns including at least
                  'Path', 'Transcript', 'IPA', 'IPA_Transcript', 'Error Category'.
        directory_root: root folder to prepend to 'Path' to make an absolute audio path.
        item_range: optional (start_idx, end_idx) inclusive. If provided, only rows in that
                    index interval (using the CSV's current order) are used.
        strip_filename: if True (default), keep only the file name (drop directories)
                        from the Path column before combining with directory_root.

    Returns:
        List of dicts with keys: {'word', 'ipa', 'audio'}.
          - 'word' is taken from 'Transcript'.
          - 'ipa' is:
               * None if 'Phonological' is NOT in 'Error Category' (per your rule)
               * Otherwise a string or list of strings parsed from 'IPA' (and optionally
                 augmented with 'IPA_Transcript' if present and distinct).
          - 'audio' is directory_root + Path (normalized).
    """
    df = pd.read_csv(csv_path)
    required_cols = ["Path", "Transcript", "Error Category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    has_ipa = "IPA" in df.columns
    has_ipa_trans = "IPA_Transcript" in df.columns

    # Apply item_range if requested
    if item_range is not None:
        if not (isinstance(item_range, (tuple, list)) and len(item_range) == 2):
            raise ValueError("item_range must be a (start_idx, end_idx) tuple or list.")
        start_idx, end_idx = item_range
        start_idx = max(0, int(start_idx))
        end_idx = min(len(df) - 1, int(end_idx))
        df = df.iloc[start_idx:end_idx + 1]

    directory_root = Path(directory_root)
    items: List[Dict] = []

    for _, row in df.iterrows():
        word = str(row.get("Transcript", "")).strip()
        rel_path = str(row.get("Path", "")).strip()
        err_cat = str(row.get("Error Category", "")).strip()

        # Optionally strip directory structure
        if strip_filename:
            rel_path = Path(rel_path).name

        # Build audio path
        audio = str((directory_root / rel_path).resolve())

        ipa_value: Optional[Union[str, List[str]]] = None

        if "Phonological" not in err_cat and "Disfluency" not in err_cat:
            ipa_value = None
        else:
            candidates: List[str] = []
            if has_ipa:
                raw_ipa = row.get("IPA")
                candidates = ipa_candidates_from_raw(raw_ipa)

            if has_ipa_trans:
                trans_ipa = row.get("IPA_Transcript")
                if isinstance(trans_ipa, str) and trans_ipa.strip():
                    extra_cands = ipa_candidates_from_raw(trans_ipa)
                    for cand in extra_cands:
                        if cand not in candidates:
                            candidates.append(cand)

            if len(candidates) == 0:
                ipa_value = None
            elif len(candidates) == 1:
                ipa_value = candidates[0]
            else:
                ipa_value = candidates

        items.append({
            "word": word,
            "ipa": ipa_value,
            "audio": audio,
        })

    return items



if __name__ == "__main__":
    # Quick demos:
    tests = [
        "deɪ",  # should become d ej
        "t͡ʃæt",  # might include tʃ if ligature was typed; our greedy tokenizer picks 'tʃ'+'æ'+'t'
        "kʰæt",  # already close to inventory
        "s.sɪli",  # stutter/cutoff: generate 's' as prefix and 's ɪ ɫ i' as base
        "ʃeɪd;seɪd",  # multiple candidates
        "oʊpən",  # oʊ -> ow
        "bɔɪ",  # ɔɪ -> ɔj
    ]
    demo = {t: ipa_candidates_from_raw(t) for t in tests}
    print(demo)


    CSV_PATH = Path("../../processed_annotations/word_level_data.csv")
    # Generate with defaults for a quick preview (no range)
    items_all = generate_mfa_items(CSV_PATH, directory_root="/", item_range=[0,10])

    # Save to JSON for you to reuse in your pipeline
    out_json = Path("../../processed_data/tmp/mfa_items.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(items_all[:100], f, ensure_ascii=False, indent=2)  # save a small sample for inspection


    print(items_all)
    print(len(items_all), str(out_json))