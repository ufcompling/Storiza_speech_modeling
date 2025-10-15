import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal

# Try to import statsmodels for linear mixed-effects
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# --- Parsing helpers ---------------------------------------------------------

# Flexible keys found in word-level CSVs
WKEY = re.compile(r"^uid_([^_]+)_sid_([^_]+)_([^_]+)$")


def parse_word_key(key: str):
    """
    Extract (uid, audio_id) from a word-level key.
    Tries a strict regex first; if it fails, falls back to underscore splitting.

    Examples of accepted keys:
      - 'uid_123_sid_456_audioABC'
      - 'someprefix_uid_123_something_audio_audioABC' (fallback logic)
    """
    s = str(key).strip()
    m = WKEY.match(s)
    if m:
        return m.group(1), m.group(3)

    parts = s.split("_")
    try:
        # User's requested indices; keep but guard against OOB
        uid = parts[4] if len(parts) > 4 else None
        audio_id = parts[7] if len(parts) > 7 else (parts[3] if len(parts) > 3 else None)
        return uid, audio_id
    except Exception:
        return None, None


def find_uid_column(meta_cols):
    """Try to find the UID column in story.xlsx."""
    candidates = [
        "userId (matches the uid in the recording file name)",
        "uid",
        "userId",
        "User ID",
        "user_id",
    ]
    for c in candidates:
        if c in meta_cols:
            return c
    # Fallback: choose the first column that looks like an ID
    for c in meta_cols:
        if "uid" in c.lower() or "user" in c.lower():
            return c
    raise ValueError("Could not find a UID column in story.xlsx")


def find_grade_column(meta_cols, exclude_col):
    """Try to find the grade/class column in story.xlsx."""
    candidates = ["grade", "Grade", "GRADE", "class", "Class"]
    for c in candidates:
        if c in meta_cols:
            return c
    # Fallback: pick the first column that's not the UID column
    for c in meta_cols:
        if c != exclude_col:
            return c
    raise ValueError("Could not find a grade/class column in story.xlsx")


# --- Core computation ---------------------------------------------------------

def read_word_level(word_path: Path, key_col_candidates=None) -> pd.DataFrame:
    """Read word-level CSV and normalize to a 'word_key' column."""
    dfw = pd.read_csv(word_path)
    if key_col_candidates is None:
        key_col_candidates = ["recording_key", "key", "utterance_key", "audio_key", "uid_sid_audio"]
    key_col = None
    for c in key_col_candidates:
        if c in dfw.columns:
            key_col = c
            break
    if key_col is None:
        # Fall back to first column
        key_col = dfw.columns[0]

    dfw = dfw.rename(columns={key_col: "word_key"}).copy()
    dfw["word_key"] = dfw["word_key"].astype(str).str.strip()
    return dfw[["word_key"]]


def normalize_grade_label(s):
    if pd.isna(s):
        return s
    s2 = str(s).strip()
    # Normalize common variants
    lowers = s2.lower()
    if "kind" in lowers:
        return "Kindergarten"
    if "1st" in lowers or "first" in lowers:
        return "1st Grade"
    if "2nd" in lowers or "second" in lowers:
        return "2nd Grade"
    # Default to 3rd+ bucket
    return "3rd Grade and Higher"


def grade_to_numeric(grade_str: str) -> int | None:
    """
    Map grade strings to numeric values:
      Kindergarten -> 0
      1st Grade -> 1
      2nd Grade -> 2
      3rd Grade and Higher -> 3
    """
    if pd.isna(grade_str):
        return None
    s = normalize_grade_label(grade_str)
    if s == "Kindergarten":
        return 0
    if s == "1st Grade":
        return 1
    if s == "2nd Grade":
        return 2
    return 3


def compute_wpm_by_grade(stories_df: pd.DataFrame,
                         meta_df: pd.DataFrame,
                         word_df: pd.DataFrame):
    """
    Returns:
      per_speaker_wpm: uid, avg_wpm_by_speaker, grade
      avg_wpm_by_grade: grade, avg_wpm, n_speakers
      w_join_with_grade: per-(uid,audio_id) obs with n_words, story_length, wpm, grade, grade_num
    """
    # Normalize key columns
    for col in ["uid", "audio_id"]:
        stories_df[col] = stories_df[col].astype(str).str.strip()
    # Expect 'story_length' in seconds
    if "story_length" not in stories_df.columns:
        raise ValueError("stories_df must contain a 'story_length' column (seconds).")

    # Parse (uid, audio_id) out of word-level keys
    word_df[["uid", "audio_id"]] = word_df["word_key"].apply(
        lambda s: pd.Series(parse_word_key(s))
    )
    word_df = word_df.dropna(subset=["uid", "audio_id"]).copy()
    word_df["uid"] = word_df["uid"].astype(str).str.strip()
    word_df["audio_id"] = word_df["audio_id"].astype(str).str.strip()

    # Word counts per (uid, audio_id)
    word_counts = (word_df
                   .groupby(["audio_id", "uid"], as_index=False)
                   .size()
                   .rename(columns={"size": "n_words"}))

    # Join with story lengths
    story_len = stories_df[["audio_id", "uid", "story_length"]].copy()
    w_join = word_counts.merge(story_len, on=["audio_id", "uid"], how="inner")

    # Keep positive durations
    w_join = w_join[w_join["story_length"] > 0].copy()

    if not w_join[w_join["story_length"] <= 0].copy().empty:
        print("WARNING: some story lengths <= 0")
        print(w_join[w_join["story_length"] <= 0].copy())

    # WPM for each (uid, audio_id)
    w_join["wpm"] = w_join["n_words"] / (w_join["story_length"] / 60.0)

    # Attach grade + normalize labels
    per_uid_grade = meta_df[["uid", "grade"]].dropna(subset=["uid"]).copy()
    per_uid_grade["uid"] = per_uid_grade["uid"].astype(str).str.strip()
    per_uid_grade["grade"] = per_uid_grade["grade"].apply(normalize_grade_label)
    w_join = w_join.merge(per_uid_grade, on="uid", how="left")
    w_join = w_join.dropna(subset=["grade"]).copy()
    w_join["grade_num"] = w_join["grade"].apply(grade_to_numeric)

    # Average WPM per speaker (uid)
    per_speaker_wpm = (w_join
                       .groupby("uid", as_index=False)["wpm"]
                       .mean()
                       .rename(columns={"wpm": "avg_wpm_by_speaker"}))
    per_speaker_wpm = per_speaker_wpm.merge(per_uid_grade, on="uid", how="left").dropna(subset=["grade"])
    per_speaker_wpm["grade_num"] = per_speaker_wpm["grade"].apply(grade_to_numeric)

    # Average WPM by grade
    avg_wpm_by_grade = (per_speaker_wpm
                        .groupby("grade", as_index=False)
                        .agg(avg_wpm=("avg_wpm_by_speaker", "mean"),
                             n_speakers=("uid", "nunique")))

    return per_speaker_wpm, avg_wpm_by_grade, w_join


def plot_stacked_histogram(per_speaker_wpm: pd.DataFrame, out_path: Path, bins: int = 20):
    """Plot a stacked histogram of per-speaker WPM by grade."""
    df = per_speaker_wpm.rename(columns={"avg_wpm_by_speaker": "wpm"})
    vals = df["wpm"].dropna().values
    if vals.size == 0:
        return
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    grouped = []
    labels = []
    for grade, g in df.groupby("grade"):
        x = g["wpm"].dropna().values
        if x.size > 0:
            grouped.append(x)
            labels.append(str(grade))

    if len(grouped) == 0:
        return

    plt.figure()
    plt.hist(grouped, bins=bin_edges, stacked=True, label=labels)
    plt.title("WPM (per speaker) — stacked by grade")
    plt.xlabel("words per minute")
    plt.ylabel("count")
    plt.legend(title="grade")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_scatter_time_words_by_grade(w_join_with_grade: pd.DataFrame, out_path: Path, *, intercept_zero: bool = True):
    """
    Scatter: time (story_length, seconds) vs total words (n_words), colored by grade.
    Overlay per-grade linear lines.
      - If intercept_zero=True, constrain lines to pass through the origin (b=0): y = m*x.
      - Otherwise, fit unrestricted y = a + b*x.
    """
    df = w_join_with_grade.dropna(subset=["story_length", "n_words", "grade"]).copy()
    df["grade"] = df["grade"].apply(normalize_grade_label)

    # Determine present grades and color map
    desired_order = ["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade and Higher"]
    present = [g for g in desired_order if g in set(df["grade"])]
    if not present:
        present = sorted(df["grade"].unique())

    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    color_map = {g: default_colors[i % len(default_colors)] for i, g in enumerate(present)}

    plt.figure()
    for g in present[:4]:
        sub = df[df["grade"] == g]
        if sub.empty:
            continue
        x = sub["story_length"].values
        y = sub["n_words"].values
        plt.scatter(x, y, label=g, s=20, alpha=0.7, color=color_map[g])

        # Trend lines
        if len(sub) >= 2:
            x_line = np.linspace(x.min(), x.max(), 100)
            if intercept_zero:
                # Fit through origin: minimize ||y - m x||, m = sum(x*y) / sum(x^2)
                denom = np.sum(x * x)
                if denom > 0:
                    m = np.sum(x * y) / denom
                    y_line = m * x_line
                    plt.plot(x_line, y_line, color=color_map[g])
            else:
                b, a = np.polyfit(x, y, 1)  # slope, intercept
                y_line = b * x_line + a
                plt.plot(x_line, y_line, color=color_map[g])

    plt.xlabel("Story length (seconds)")
    plt.ylabel("Total words")
    plt.title("Time vs Total Words by Grade" + (" (lines through origin)" if intercept_zero else ""))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Also emit a small summary to help debug "no blue points"
    counts = df.groupby("grade").size().rename("n_points").reset_index()
    counts_path = Path(out_path).with_suffix(".counts.tsv")
    counts.to_csv(counts_path, sep="\t", index=False)


def make_boxplot_wpm_by_grade(per_speaker_wpm: pd.DataFrame, out_path: Path) -> dict:
    """
    Box-and-whisker plot for per-speaker WPM by grade.
    Returns a dict with significance test results (Spearman, Kruskal).
    """
    order = ["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade and Higher"]
    # Keep only grades we actually have, preserving order
    present = [g for g in order if g in set(per_speaker_wpm["grade"])]
    data = [per_speaker_wpm.loc[per_speaker_wpm["grade"] == g, "avg_wpm_by_speaker"].dropna().values for g in present]

    plt.figure()
    # Use tick_labels to avoid Matplotlib 3.9 deprecation warning
    plt.boxplot(data, tick_labels=present, showfliers=True)
    plt.ylabel("Avg WPM by speaker")
    plt.title("WPM by Grade (Per Speaker)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Spearman rank correlation (grade vs per-speaker WPM)
    tmp = per_speaker_wpm.dropna(subset=["grade", "avg_wpm_by_speaker"]).copy()
    tmp["grade_num"] = tmp["grade"].apply(grade_to_numeric)
    rho, p_spear = spearmanr(tmp["grade_num"].values, tmp["avg_wpm_by_speaker"].values)

    # Kruskal-Wallis across grade groups
    kruskal_groups = [d for d in data if len(d) > 0]
    if len(kruskal_groups) >= 2:
        H, p_kruskal = kruskal(*kruskal_groups)
    else:
        H, p_kruskal = np.nan, np.nan

    return {
        "spearman_rho": rho,
        "spearman_p": p_spear,
        "kruskal_H": H,
        "kruskal_p": p_kruskal,
    }


def fit_lme_wpm(w_join_with_grade: pd.DataFrame, out_tsv: Path) -> pd.DataFrame:
    """
    Fit a linear mixed-effects model:
      wpm ~ grade_num + n_words  with random intercepts by uid
    Saves fixed-effects summary to TSV and returns it.
    Also writes variance components (if available) to a separate file.
    """
    if not HAS_STATSMODELS:
        print("statsmodels not available; skipping LME fit.")
        return pd.DataFrame()

    df = w_join_with_grade.dropna(subset=["wpm", "grade_num", "n_words", "uid"]).copy()
    # Ensure numeric
    df["grade_num"] = pd.to_numeric(df["grade_num"], errors="coerce")
    df["n_words"] = pd.to_numeric(df["n_words"], errors="coerce")
    df = df.dropna(subset=["grade_num", "n_words"]).copy()

    # Fit MixedLM with random intercept by uid
    try:
        model = smf.mixedlm("wpm ~ grade_num + n_words", df, groups=df["uid"])
        result = model.fit(reml=False, method="lbfgs")
    except Exception as e:
        print(f"MixedLM failed to fit: {e}")
        return pd.DataFrame()

    # Align on the fixed-effect index to avoid length mismatches.
    fe_index = result.fe_params.index
    fe_params = result.fe_params.reindex(fe_index)
    fe_se = result.bse_fe.reindex(fe_index)
    fe_p = result.pvalues.reindex(fe_index)  # drop entries like 'Group Var' that appear in pvalues

    fe_z = fe_params / fe_se

    out = pd.DataFrame({
        "term": fe_index,
        "coef": fe_params.values,
        "std_err": fe_se.values,
        "z_value": fe_z.values,
        "p_value": fe_p.values
    })
    out.to_csv(out_tsv, sep="\t", index=False)

    # Save variance components (if exposed)
    varcomp_path = Path(str(out_tsv).replace("_fixed_effects.tsv", "_variance_components.tsv"))
    try:
        vc = []
        if hasattr(result, "random_effects") and len(result.random_effects) > 0:
            # random intercept variance
            if hasattr(result, "cov_re"):
                vc.append({"component": "Group Var", "value": float(np.squeeze(np.asarray(result.cov_re))) })
        if hasattr(result, "scale"):
            vc.append({"component": "Residual Var", "value": float(result.scale)})
        if vc:
            pd.DataFrame(vc).to_csv(varcomp_path, sep="\t", index=False)
    except Exception:
        pass

    print("\n=== MixedLM Fixed Effects (wpm ~ grade_num + n_words) ===")
    print(out)

    return out


# --- Main (parameterized) ----------------------------------------------------

def main(*,
         processed_dir: Path | str | None = None,
         stories: Path | str | None = None,
         meta: Path | str | None = None,
         words: Path | str | None = None,
         out_dir: Path | str | None = None,
         no_plot: bool = False):
    """
    Run the WPM by grade pipeline with explicit parameters.

    Provide either:
      - processed_dir: directory containing story_lengths.tsv, story.xlsx, word_level_data.csv
        (other specific paths may be left as None)
    OR
      - stories, meta, words: explicit paths to each file

    Args:
      processed_dir: base directory with input files (optional if explicit paths are given)
      stories: path to story_lengths.tsv
      meta: path to story.xlsx
      words: path to word_level_data.csv
      out_dir: where outputs will be written (defaults to processed_dir or stories' parent)
      no_plot: if True, skip saving the histogram and other plots
    """
    # Resolve input paths
    if processed_dir is not None:
        base = Path(processed_dir)
        stories_path = Path(stories) if stories else base / "story_lengths.tsv"
        meta_path = Path(meta) if meta else base / "story.xlsx"
        words_path = Path(words) if words else base / "word_level_data.csv"
        out_path = Path(out_dir) if out_dir else base
    else:
        if not (stories and meta and words):
            raise ValueError("If processed_dir is not provided, you must specify stories, meta, and words.")
        stories_path = Path(stories)
        meta_path = Path(meta)
        words_path = Path(words)
        out_path = Path(out_dir) if out_dir else stories_path.parent

    out_path.mkdir(parents=True, exist_ok=True)

    # Read inputs
    stories_df = pd.read_csv(stories_path, sep="\t")
    required_story_cols = {"uid", "audio_id", "story_length"}
    missing = required_story_cols - set(stories_df.columns)
    if missing:
        raise ValueError(f"story_lengths.tsv is missing columns: {sorted(missing)}")

    meta_raw = pd.read_excel(meta_path)
    uid_col = find_uid_column(meta_raw.columns)
    grade_col = find_grade_column(meta_raw.columns, exclude_col=uid_col)
    meta_df = (meta_raw[[uid_col, grade_col]]
               .rename(columns={uid_col: "uid", grade_col: "grade"}))
    meta_df["uid"] = meta_df["uid"].astype(str).str.strip()

    word_df = read_word_level(words_path)

    # Compute
    per_speaker_wpm, avg_wpm_by_grade, w_join = compute_wpm_by_grade(
        stories_df=stories_df,
        meta_df=meta_df,
        word_df=word_df
    )

    # Save tables
    avg_path = out_path / "avg_wpm_by_grade.tsv"
    speaker_path = out_path / "per_speaker_wpm.tsv"
    avg_wpm_by_grade.to_csv(avg_path, sep="\t", index=False)
    per_speaker_wpm.to_csv(speaker_path, sep="\t", index=False)

    # Optional plots
    if not no_plot:
        fig_hist = out_path / "stacked_hist_wpm_by_grade.png"
        plot_stacked_histogram(per_speaker_wpm, fig_hist, bins=20)
        print(f"Saved histogram to: {fig_hist}")

        fig_scatter = out_path / "scatter_time_vs_total_words_by_grade.png"
        make_scatter_time_words_by_grade(w_join, fig_scatter, intercept_zero=True)  # constrain b=0
        print(f"Saved scatter (time vs words) to: {fig_scatter}")

        fig_box = out_path / "boxplot_wpm_by_grade.png"
        sig = make_boxplot_wpm_by_grade(per_speaker_wpm, fig_box)
        print(f"Saved boxplot to: {fig_box}")

        # Save significance test results
        sig_df = pd.DataFrame([sig])
        sig_path = out_path / "wpm_by_grade_significance.tsv"
        sig_df.to_csv(sig_path, sep="\t", index=False)
        print(f"Saved significance tests to: {sig_path}")
    else:
        sig = {}

    # Fit linear mixed effects model
    lme_path = out_path / "lme_wpm_fixed_effects.tsv"
    lme_df = fit_lme_wpm(w_join, lme_path)

    # Console summary
    print(f"Saved avg WPM by grade to: {avg_path}")
    print(f"Saved per-speaker WPM to: {speaker_path}")
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print("\n=== Avg WPM by Grade ===")
        print(avg_wpm_by_grade.sort_values('grade'))

    # Print brief stats summary
    if sig:
        print("\n=== Significance (Per Speaker) ===")
        print(pd.DataFrame([sig]))


# --- Predefined parameters (edit here) ---------------------------------------

if __name__ == "__main__":
#    ROOT_DIR = os.path.join("..", "..")
#    PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_data")
#    PROCESSED_ANNOTATON_DIR = os.path.join(ROOT_DIR, "processed_annotations")
    PROCESSED_DIR = ("processed_annotations")

    STORIES = Path(os.path.join(PROCESSED_DIR, "story_lengths.tsv"))
    META = Path(os.path.join(PROCESSED_DIR, "story.xlsx"))
    WORDS = Path(os.path.join(PROCESSED_DIR, "word_level_data.csv"))
    OUT_DIR = Path(os.path.join(PROCESSED_DIR, "wpm_outputs"))
    NO_PLOT = False

    # Ensure base directory exists (won't fail if it already does)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run with the predefined parameters
    main(
        processed_dir=PROCESSED_DIR,
        stories=STORIES,
        meta=META,
        words=WORDS,
        out_dir=OUT_DIR,
        no_plot=NO_PLOT,
    )