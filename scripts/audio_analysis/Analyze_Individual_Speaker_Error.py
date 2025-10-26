#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate MFA per-UID JSONs and visualize alignment quality metrics:
1. Box-and-whisker plots for speech_log_likelihood, overall_log_likelihood, and snr.
2. Histograms for each metric.
3. Histograms of per-speaker averages of each metric.
Adds a red threshold line for log-likelihoods below -70.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_uid_json(fp: Path) -> pd.DataFrame:
    """Load one {uid}.json into a pandas DataFrame."""
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    uid = data.get("uid")
    rows = []
    for it in data.get("items", []):
        rows.append({
            "uid": uid,
            "speaker": it.get("speaker"),
            "word": it.get("word"),
            "audio": it.get("audio"),
            "best_ipa": it.get("best_ipa"),
            "speech_log_likelihood": it.get("speech_log_likelihood"),
            "overall_log_likelihood": it.get("overall_log_likelihood"),
            "snr": it.get("snr"),
        })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, outdir: Path, threshold: float = -70.0):
    """Generate boxplots, histograms, and per-speaker histograms."""
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = ["speech_log_likelihood", "overall_log_likelihood", "snr"]
    df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")

    # Count how many items fall below threshold
    for m in ["speech_log_likelihood", "overall_log_likelihood"]:
        below = df[m] < threshold
        percent = below.mean(skipna=True) * 100
        print(f"{m}: {below.sum()} of {below.count()} items ({percent:.2f}%) below {threshold}")
    print()

    # 1) Box-and-whisker plot (matplotlib, not seaborn)
    # Use plot.box so we can access artists for coloring
    ax = df[metrics].plot.box(patch_artist=True)
    ax.set_title("Box & Whisker: Speech/Overall Log-Likelihood and SNR")
    ax.set_ylabel("Value")

    # ax.artists corresponds to boxes in the same order as 'metrics'
    for patch, label in zip(ax.artists, metrics):
        if "likelihood" in label:
            color = "red" if df[label].mean() < threshold else "lightblue"
        else:
            color = "lightblue"
        patch.set_facecolor(color)

    # Add red dashed threshold line for log-likelihoods
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_ll_snr.png", bbox_inches="tight", dpi=200)
    plt.close()

    # 2) Histograms per metric
    for m in metrics:
        plt.figure()
        series = df[m].dropna()
        if "likelihood" in m:
            # Items below threshold in red, above in gray
            below = series[series < threshold]
            above = series[series >= threshold]
            plt.hist([above, below], bins=50, stacked=True,
                     color=["gray", "red"], label=[f">= {threshold}", f"< {threshold}"])
            plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5)
            plt.legend()
        else:
            plt.hist(series, bins=50, color="gray")

        plt.title(f"Histogram: {m}")
        plt.xlabel(m)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_{m}.png", bbox_inches="tight", dpi=200)
        plt.close()

    # 3) Per-speaker averages + histograms
    who = df["speaker"].fillna(df["uid"])
    df["who"] = who
    per_speaker = (
        df.groupby("who")[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"who": "speaker"})
    )

    per_speaker.to_csv(outdir / "per_speaker_averages.csv", index=False)

    for m in metrics:
        plt.figure()
        series = per_speaker[m].dropna()
        plt.hist(series, bins=30, color="gray")
        plt.title(f"Histogram of per-speaker mean: {m}")
        plt.xlabel(f"mean({m})")
        plt.ylabel("Speakers")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_per_speaker_mean_{m}.png", bbox_inches="tight", dpi=200)
        plt.close()


if __name__ == "__main__":
    # === CONFIGURE THESE PATHS ===
    JSON_DIR = Path("../../processed_data/phoneme_segmentation_by_speaker")
    OUT_DIR = Path("../../processed_data/figures")
    THRESHOLD = -65.0  # Red dashed line for log-likelihood cutoff

    # Ensure OUT_DIR exists before writing CSV/figures
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === LOAD ALL JSONS ===
    json_files = sorted(JSON_DIR.glob("uid_*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {JSON_DIR}")

    dfs = [load_uid_json(p) for p in json_files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(OUT_DIR / "all_items.csv", index=False)

    # === MAKE PLOTS ===
    make_plots(df_all, OUT_DIR, threshold=THRESHOLD)

    print(f"\n✅ Done. Saved plots and CSVs to: {OUT_DIR.resolve()}")
    print(f"  Total items: {len(df_all)} | Speakers: {df_all['speaker'].nunique()}")
