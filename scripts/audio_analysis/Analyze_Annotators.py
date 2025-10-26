#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize MFA alignment quality by ANNOTATOR.

- Extracts annotation IDs from audio filenames
- Maps them to annotator names via the provided export CSV
- Creates boxplots for overall_log_likelihood and speech_log_likelihood
- Computes per-annotator averages and prints/saves them in descending order
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Helpers
# -------------------------------

ANNOTATION_ID_RE = re.compile(r"_sid_[^_]+_\d+_(\d+)_word_segment_", re.IGNORECASE)

def extract_annotation_id(path_str: str) -> str:
    """Extract the numeric annotation ID (e.g., 195799964) from the audio filename."""
    m = ANNOTATION_ID_RE.search(str(path_str))
    return m.group(1) if m else None


def load_uid_json(fp: Path) -> pd.DataFrame:
    """Load a single UID JSON into a pandas DataFrame."""
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    uid = data.get("uid")
    rows = []
    for it in data.get("items", []):
        audio = it.get("audio")
        rows.append({
            "uid": uid,
            "annotation_id": extract_annotation_id(audio),
            "speaker": it.get("speaker"),
            "word": it.get("word"),
            "audio": audio,
            "best_ipa": it.get("best_ipa"),
            "speech_log_likelihood": it.get("speech_log_likelihood"),
            "overall_log_likelihood": it.get("overall_log_likelihood"),
            "snr": it.get("snr"),
        })
    return pd.DataFrame(rows)


def load_annotation_lookup(csv_path: Path) -> pd.DataFrame:
    """Load the project export CSV and return a {annotation_id → annotator_name} mapping."""
    # If your file is TSV, keep delimiter="\t"; if CSV, remove it.
    df = pd.read_csv(csv_path, dtype=str, delimiter="\t")
    possible_annot_cols = [c for c in df.columns if "annotator" in c.lower()]
    if not possible_annot_cols:
        raise RuntimeError("Could not find an 'annotator' column in the export file.")
    id_col = "id"  # adjust if your export uses a different name
    annot_col = possible_annot_cols[0]
    mapping = df[[id_col, annot_col]].dropna().drop_duplicates()
    mapping.columns = ["annotation_id", "annotator"]
    return mapping


# -------------------------------
# Plotting
# -------------------------------

def boxplot_by_annotator(df: pd.DataFrame, metric: str, out_png: Path, threshold=-70.0):
    df = df[["annotator", metric]].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    if df.empty:
        print(f"[WARN] No valid data for {metric}.")
        return

    print(f"\n=== {metric} by annotator ===")
    for ann, sub in df.groupby("annotator"):
        pct_below = (sub[metric] < threshold).mean() * 100
        print(f"{ann}: {pct_below:.2f}% below {threshold} (n={len(sub)})")

    ax = df.boxplot(column=metric, by="annotator", patch_artist=True, rot=90)
    ax.set_title(f"{metric} by annotator")
    ax.set_ylabel(metric)
    ax.set_xlabel("Annotator")

    medians = df.groupby("annotator")[metric].median()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]

    for patch, annot in zip(ax.artists, xticklabels):
        median = medians.get(annot)
        color = "red" if median is not None and median < threshold else "lightblue"
        patch.set_facecolor(color)

    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    ax.legend()
    plt.tight_layout()
    fig = ax.get_figure()
    fig.suptitle("")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


# -------------------------------
# Stats by annotator
# -------------------------------

def summarize_by_annotator(df_all: pd.DataFrame, out_csv: Path, threshold: float):
    # Ensure numeric
    df = df_all[["annotator", "overall_log_likelihood", "speech_log_likelihood"]].copy()
    df["overall_log_likelihood"] = pd.to_numeric(df["overall_log_likelihood"], errors="coerce")
    df["speech_log_likelihood"]  = pd.to_numeric(df["speech_log_likelihood"],  errors="coerce")

    g = df.groupby("annotator", dropna=False)

    # Build the table column-by-column (avoids agg syntax pitfalls)
    stats = pd.DataFrame({
        "n":                       g["overall_log_likelihood"].count(),
        "mean_overall":            g["overall_log_likelihood"].mean(),
        "mean_speech":             g["speech_log_likelihood"].mean(),
        "pct_overall_below":       g["overall_log_likelihood"].apply(lambda s: (s < threshold).mean() * 100),
        "pct_speech_below":        g["speech_log_likelihood"].apply(lambda s: (s < threshold).mean() * 100),
    })

    # Print sorted by mean_overall (descending: less negative = better)
    print("\n=== Mean overall_log_likelihood by annotator (descending) ===")
    for ann, row in stats.sort_values("mean_overall", ascending=False).iterrows():
        print(f"{ann:30s} mean_overall={row['mean_overall']:.2f}  n={int(row['n'])}  "
              f"%overall_below({threshold})={row['pct_overall_below']:.2f}%")

    # Print sorted by mean_speech (descending)
    print("\n=== Mean speech_log_likelihood by annotator (descending) ===")
    for ann, row in stats.sort_values("mean_speech", ascending=False).iterrows():
        print(f"{ann:30s} mean_speech={row['mean_speech']:.2f}  n={int(row['n'])}  "
              f"%speech_below({threshold})={row['pct_speech_below']:.2f}%")

    stats.reset_index().to_csv(out_csv, index=False)
    print(f"\nSaved per-annotator summary to: {out_csv}")


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    # === CONFIG ===
    JSON_DIR = Path("../../processed_data/phoneme_segmentation_by_speaker")
    ANNOTATION_CSV = Path("../../processed_annotations/export_157618_project-157618-at-2025-10-15-13-11-70ea5d48.csv")
    OUT_DIR = Path("../../processed_data/figures_by_annotator")
    THRESHOLD = -65.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotation ID → annotator name mapping
    lookup = load_annotation_lookup(ANNOTATION_CSV)

    # Load all JSONs
    json_files = sorted(JSON_DIR.glob("uid_*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {JSON_DIR}")

    dfs = [load_uid_json(p) for p in json_files]
    df_all = pd.concat(dfs, ignore_index=True)

    # Merge annotator info
    df_all = df_all.merge(lookup, on="annotation_id", how="left")
    df_all["annotator"] = df_all["annotator"].fillna("UNKNOWN")

    # Persist joined items (optional)
    df_all.to_csv(OUT_DIR / "all_items_with_annotators.csv", index=False)

    # Make plots
    boxplot_by_annotator(df_all, "overall_log_likelihood",
                         OUT_DIR / "box_overall_by_annotator.png", threshold=THRESHOLD)
    boxplot_by_annotator(df_all, "speech_log_likelihood",
                         OUT_DIR / "box_speech_by_annotator.png", threshold=THRESHOLD)

    # Print & save per-annotator averages (descending)
    summarize_by_annotator(
        df_all,
        out_csv=OUT_DIR / "per_annotator_summary.csv",
        threshold=THRESHOLD,
    )

    print(f"\n✅ Saved per-annotator boxplots to: {OUT_DIR.resolve()}")
