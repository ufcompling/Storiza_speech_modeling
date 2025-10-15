#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

# Local imports (adjust paths/names if your modules live elsewhere)
from MFA_API import MFAIpaAligner
from MFA_Audio_Preprocesser import generate_mfa_items


# -------------------------
# Helpers for UID grouping
# -------------------------

UID_RE = re.compile(r"(uid_[A-Za-z0-9]+)")

def extract_uid(path_str: Union[str, Path]) -> Optional[str]:
    """Return first 'uid_...' token from a path, else None."""
    if not path_str:
        return None
    m = UID_RE.search(str(path_str))
    return m.group(1) if m else None


def write_jsons_by_uid(
    results: List[Dict],
    out_dir: Union[str, Path] = "../../processed_data/phoneme_segmentation_by_speaker",
) -> Dict[str, int]:
    """
    Group aligner results by UID and write one JSON per UID with:
      - word
      - best_ipa (string)
      - best_ipa_phones (list[str])  # derived from phones if available, else split best_ipa
      - phones: [{phone, start, end}]  # per-phoneme timings from MFA
      - audio (original audio path if available, else aligned copy)
      - speaker
      - overall_log_likelihood
      - speech_log_likelihood
      - phone_duration_deviation
      - snr

    Returns: {uid: count_of_items}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict]] = defaultdict(list)

    EXTRA_KEYS = [
        "speaker",
        "overall_log_likelihood",
        "speech_log_likelihood",
        "phone_duration_deviation",
        "snr",
    ]

    for r in results:
        # Prefer original source audio if present (so UID is discoverable)
        audio_for_uid = r.get("orig_audio") or r.get("audio")
        uid = extract_uid(audio_for_uid) if audio_for_uid else None
        if not uid:
            uid = "_NO_UID_"

        # Per-phone segments straight from MFA:
        segs = r.get("phone_segments") or []
        # Normalize to a simple schema
        phones = [
            {
                "phone": ph.get("phone"),
                "start": float(ph.get("start")) if ph.get("start") is not None else None,
                "end": float(ph.get("end")) if ph.get("end") is not None else None,
            }
            for ph in segs
        ]

        # Derive best_ipa_phones from segments if present; otherwise split best_ipa string
        if phones:
            best_ipa_phones = [ph["phone"] for ph in phones if ph.get("phone")]
        else:
            best_ipa_phones = (r.get("best_ipa") or "").split()

        item = {
            "word": r.get("word"),
            "best_ipa": r.get("best_ipa"),
            "best_ipa_phones": best_ipa_phones,
            "phones": phones,                 # <-- start/end per phoneme
            "audio": audio_for_uid,
        }
        for k in EXTRA_KEYS:
            item[k] = r.get(k)

        grouped[uid].append(item)

    summary: Dict[str, int] = {}
    for uid in tqdm(sorted(grouped.keys()), desc="Writing per-speaker JSON"):
        payload = {"uid": uid, "items": grouped[uid]}
        out_path = out_dir / f"{uid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        summary[uid] = len(grouped[uid])

    return summary




# -------------------------
# Main
# -------------------------

def main():
    # --- Configure the aligner ---
    # IMPORTANT: `dictionary` should be a *file path* to a plain-text lexicon if you want
    # the batch combiner to append the base dictionary. (No `export_dictionary` needed.)
    aligner = MFAIpaAligner(
        acoustic_model="english_mfa",
        dictionary="/home/michaelbennie/Documents/MFA/pretrained_models/dictionary/english_mfa.dict",
        mfa_bin="mfa",
        temp_root="./temp",
        speaker_characters=36,         # exact length of your UID prefix
        keep_original_filenames=True,  # keep original basenames to preserve UID stems
    )

    # --- Build items from your CSV ---
    CSV_PATH = Path("../../processed_annotations/word_level_data.csv")
    audio_root = Path(
        "~/OneDrive/Leite,Walter's files - Storiza Corpus Spring 2025/Processed_Utterances/annotated_word_segments"
    ).expanduser()

    items = generate_mfa_items(
        csv_path=CSV_PATH,
        directory_root=audio_root,
        item_range=None,
        strip_filename=True          # since CSV paths already include filenames
    )

    # --- Run alignment (batch) ---
    out_align_dir = Path("./aligned_outputs")
    results = aligner.align(items, output_dir=out_align_dir)

    # --- Backfill orig_audio into results if the class didn't include it ---
    # Map original audio stems -> full original paths from items
    stem_to_orig_audio: Dict[str, str] = {
        Path(it["audio"]).stem: str(it["audio"]) for it in items if "audio" in it
    }
    for r in results:
        # The 'audio' returned by align() is usually the aligned copy inside out_dir.
        # Use its stem to find the original input path.
        if not r.get("orig_audio"):
            stem = Path(r.get("audio", "")).stem
            if stem in stem_to_orig_audio:
                r["orig_audio"] = stem_to_orig_audio[stem]

    # --- Write per-UID JSON files with a progress bar ---
    per_uid_out_dir = "../../processed_data/phoneme_segmentation_by_speaker"
    summary = write_jsons_by_uid(results, out_dir=per_uid_out_dir)

    # --- Console summary ---
    total_items = sum(summary.values())
    print(f"\nWrote {total_items} items across {len(summary)} UID files -> {per_uid_out_dir}")
    for uid, count in sorted(summary.items()):
        print(f"  {uid}: {count} items")


if __name__ == "__main__":
    main()
