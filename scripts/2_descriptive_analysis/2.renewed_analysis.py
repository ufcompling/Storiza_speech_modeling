
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATTERN = re.compile(r"^uid_([^_]+)_sid_([^_]+)_([^_]+)_([^_]+)_sentence_segment_close\.wav$")

WKEY = re.compile(r"^uid_([^_]+)_sid_([^_]+)_([^_]+)$")

def parse_filename(filename: str):
    audio_name = filename.split('/')[-1]
    m = PATTERN.match(str(audio_name).strip())
    if not m:
        return None, None
    uid = m.group(1)
    audio_id = m.group(3)
    print(uid, audio_id)
    return uid, audio_id

def parse_word_key(key: str):
    key = key.split('/')[-1]
    s = str(key).strip()
    m = WKEY.match(s)
    if m:
        return m.group(1), m.group(3)
    
    parts = s.split("_")
    try:
        uid = parts[1]
        audio_id = parts[5] if len(parts) > 5 else parts[3]
        return uid, audio_id
    except Exception:
        return None, None

def plot_stacked_histogram(df, value_col, grade_col, out_path, bins=20):
    vals = df[value_col].dropna().values
    if vals.size == 0:
        return
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    grouped = []
    labels = []
    for grade, g in df.groupby(grade_col):
        x = g[value_col].dropna().values
        if x.size > 0:
            grouped.append(x)
            labels.append(str(grade))

    if len(grouped) == 0:
        return

    plt.figure()
    plt.hist(grouped, bins=bin_edges, stacked=True, label=labels)
    plt.title(f"{value_col} stacked histogram by grade")
    plt.xlabel(value_col)
    plt.ylabel("count")
    plt.legend(title=grade_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def total_time_main(in_path: Path, out_story_path: Path, out_sum_path: Path,
                    story_excel_path: Path,
                    out_grade_means_path: Path,
                    out_grade_counts_path: Path,
                    out_stats_path: Path,
                    figs_dir: Path):
    
    df = pd.read_csv(in_path, sep=",")
    required = {"utterance_output_filename", "close_utterance_start_time", "close_utterance_end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    
    df[["uid", "audio_id"]] = df["utterance_output_filename"].apply(
        lambda s: pd.Series(parse_filename(s))
    #    lambda s: pd.Series(parse_word_key(s))
    )

    df = df.dropna(subset=["uid", "audio_id"]).copy()

    
    for c in ["close_utterance_start_time", "close_utterance_end_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close_utterance_start_time", "close_utterance_end_time"]).copy()

    
    df["utterance_duration"] = df["close_utterance_end_time"] - df["close_utterance_start_time"]

    
    story_window = df.groupby("audio_id").agg(
        earliest_start=("close_utterance_start_time", "min"),
        latest_end=("close_utterance_end_time", "max"),
        n_utterances=("audio_id", "size"),
        uid=("uid", "first"),
    )
    story_window["story_length"] = story_window["latest_end"] - story_window["earliest_start"]
    stories = story_window.reset_index()[["audio_id", "uid", "earliest_start", "latest_end", "story_length", "n_utterances"]]

    sum_durations = df.groupby(["audio_id", "uid"], as_index=False)["utterance_duration"].sum()
    sum_durations = sum_durations.rename(columns={"utterance_duration": "total_duration"})

    
    avg_story_length = stories["story_length"].mean()
    avg_sum_durations = sum_durations["total_duration"].mean()

    
    stories.to_csv(out_story_path, sep="\t", index=False)
    sum_durations.to_csv(out_sum_path, sep="\t", index=False)

    
    meta = pd.read_excel(story_excel_path)
    col_user = "userId (matches the uid in the recording file name)"
    if col_user not in meta.columns:
        raise ValueError(f"missing column in story.xlsx: {col_user}")

    grade_col = None
    for cand in ["grade", "Grade", "GRADE", "class", "Class"]:
        if cand in meta.columns:
            grade_col = cand
            break
    if grade_col is None:
        grade_col = [c for c in meta.columns if c != col_user][0]

    meta = meta[[col_user, grade_col]].rename(columns={col_user: "uid", grade_col: "grade"}).dropna(subset=["uid"])

    
    per_speaker_story = stories.groupby("uid", as_index=False)["story_length"].mean().rename(
        columns={"story_length": "avg_story_length_by_speaker"}
    )
    per_speaker_sum = sum_durations.groupby("uid", as_index=False)["total_duration"].mean().rename(
        columns={"total_duration": "avg_summed_utterance_length_by_speaker"}
    )

    speaker_metrics = per_speaker_story.merge(per_speaker_sum, on="uid", how="outer")
    speaker_metrics = speaker_metrics.merge(meta, on="uid", how="left")
    speaker_metrics = speaker_metrics.dropna(subset=["grade"]).copy()

    
    grade_means = speaker_metrics.groupby("grade").agg(
        avg_story_length_by_speaker=("avg_story_length_by_speaker", "mean"),
        avg_summed_utterance_length_by_speaker=("avg_summed_utterance_length_by_speaker", "mean"),
        n_speakers=("uid", "nunique"),
    ).reset_index()
    grade_means.to_csv(out_grade_means_path, sep="\t", index=False)

    
    grade_counts = speaker_metrics.groupby("grade").agg(n_speakers=("uid", "nunique")).reset_index()
    grade_counts.to_csv(out_grade_counts_path, sep="\t", index=False)

    
    stats_rows = []

    groups_story = [g["avg_story_length_by_speaker"].dropna().values
                    for _, g in speaker_metrics.groupby("grade")]
    if all(len(x) > 0 for x in groups_story) and len(groups_story) > 1:
        H1, p1 = kruskal(*groups_story)
        stats_rows.append({"test": "Kruskal avg_story_length_by_speaker across grades", "stat": H1, "p_value": p1})
    else:
        stats_rows.append({"test": "Kruskal avg_story_length_by_speaker across grades", "stat": np.nan, "p_value": np.nan})

    groups_sum = [g["avg_summed_utterance_length_by_speaker"].dropna().values
                  for _, g in speaker_metrics.groupby("grade")]
    if all(len(x) > 0 for x in groups_sum) and len(groups_sum) > 1:
        H2, p2 = kruskal(*groups_sum)
        stats_rows.append({"test": "Kruskal avg_summed_utterance_length_by_speaker across grades", "stat": H2, "p_value": p2})
    else:
        stats_rows.append({"test": "Kruskal avg_summed_utterance_length_by_speaker across grades", "stat": np.nan, "p_value": np.nan})

    def chi2_on_binned(series, grades, label):
        s = pd.Series(series.values, index=series.index)
        try:
            bins = pd.qcut(s, q=min(4, max(2, s.nunique())), duplicates="drop")
        except ValueError:
            return {"test": f"Chi2 {label} by grade", "chi2": np.nan, "p_value": np.nan, "dof": np.nan}
        tmp = pd.DataFrame({"bin": bins, "grade": grades})
        ct = pd.crosstab(tmp["grade"], tmp["bin"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(ct)
            return {"test": f"Chi2 {label} by grade", "chi2": chi2, "p_value": p, "dof": dof}
        else:
            return {"test": f"Chi2 {label} by grade", "chi2": np.nan, "p_value": np.nan, "dof": np.nan}

    stats_rows.append(chi2_on_binned(
        speaker_metrics["avg_story_length_by_speaker"],
        speaker_metrics["grade"],
        "avg_story_length_by_speaker"
    ))
    stats_rows.append(chi2_on_binned(
        speaker_metrics["avg_summed_utterance_length_by_speaker"],
        speaker_metrics["grade"],
        "avg_summed_utterance_length_by_speaker"
    ))

    counts = grade_counts["n_speakers"].values.astype(float)
    if len(counts) > 0:
        expected = np.repeat(counts.mean(), len(counts))
        chi2_g, p_g = chi2_contingency([counts, expected])[0:2]
        stats_rows.append({"test": "Chi2 grade speaker counts equal proportions", "chi2": chi2_g, "p_value": p_g, "dof": len(counts) - 1})
    else:
        stats_rows.append({"test": "Chi2 grade speaker counts equal proportions", "chi2": np.nan, "p_value": np.nan, "dof": np.nan})

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_stats_path, sep="\t", index=False)

    
    os.makedirs(figs_dir, exist_ok=True)
    plot_stacked_histogram(
        speaker_metrics, "avg_story_length_by_speaker", "grade",
        os.path.join(figs_dir, "stacked_hist_avg_story_length_by_speaker.png"), bins=3
    )
    plot_stacked_histogram(
        speaker_metrics, "avg_summed_utterance_length_by_speaker", "grade",
        os.path.join(figs_dir, "stacked_hist_avg_summed_utterance_length_by_speaker.png"), bins=3
    )

    
    print(f"saved story lengths to: {out_story_path}")
    print(f"saved per story summed durations to: {out_sum_path}")
    print(f"saved grade means to: {out_grade_means_path}")
    print(f"saved grade counts to: {out_grade_counts_path}")
    print(f"saved stats to: {out_stats_path}")
    print(f"saved histograms under: {figs_dir}")
    print(f"average story length seconds: {avg_story_length:.2f}")
    print(f"average sum of utterance durations seconds: {avg_sum_durations:.2f}")
    print(f"total number of stories: {len(stories)}")
    print(f"total number of speakers with grade: {speaker_metrics['uid'].nunique()}")



def read_word_level(word_path: Path, key_col_candidates=None, error_col_candidates=None):
    dfw = pd.read_csv(word_path)
    if key_col_candidates is None:
        key_col_candidates = ["recording_key", "key", "utterance_key", "audio_key", "uid_sid_audio"]
    key_col = None
    for c in key_col_candidates:
        if c in dfw.columns:
            key_col = c
            break
    if key_col is None:
        key_col = dfw.columns[0]

    if error_col_candidates is None:
        error_col_candidates = ["Error Cateorgy", "Error Category", "error_category", "error", "Error"]
    err_col = None
    for c in error_col_candidates:
        if c in dfw.columns:
            err_col = c
            break

    dfw = dfw.rename(columns={key_col: "word_key"})
    if err_col:
        dfw = dfw.rename(columns={err_col: "error_cat"})
    else:
        dfw["error_cat"] = np.nan

    dfw["word_key"] = dfw["word_key"].astype(str).str.strip()

    return dfw

def build_wpm_tables(word_df: pd.DataFrame,
                     stories_df: pd.DataFrame,
                     meta_df: pd.DataFrame,
                     out_avg_wpm_by_grade: Path,
                     out_err_table: Path,
                     out_err_stats: Path,
                     figs_dir: Path):
    
    word_df[["uid", "audio_id"]] = word_df["word_key"].apply(lambda s: pd.Series(parse_word_key(s)))
    word_df = word_df.dropna(subset=["uid", "audio_id"]).copy()

    word_df["uid"] = word_df["uid"].astype(str).str.strip()
    word_df["audio_id"] = word_df["audio_id"].astype(str).str.strip()

    stories_df = stories_df.copy()
    stories_df["uid"] = stories_df["uid"].astype(str).str.strip()
    stories_df["audio_id"] = stories_df["audio_id"].astype(str).str.strip()
    
    word_counts = word_df.groupby(["audio_id", "uid"], as_index=False).size().rename(columns={"size": "n_words"})

    story_len = stories_df[["audio_id", "uid", "story_length"]].copy()

    w_join = word_counts.merge(story_len, on=["audio_id", "uid"], how="inner")
    
    w_join = w_join[w_join["story_length"] > 0].copy()

    
    w_join["wpm"] = w_join["n_words"] / (w_join["story_length"] / 60.0)

    
    per_speaker_wpm = w_join.groupby("uid", as_index=False)["wpm"].mean().rename(columns={"wpm": "avg_wpm_by_speaker"})

    
    speaker_grade = meta_df[["uid", "grade"]].dropna(subset=["uid"]).copy()
    per_speaker_wpm = per_speaker_wpm.merge(speaker_grade, on="uid", how="left").dropna(subset=["grade"])

    
    avg_wpm_by_grade = per_speaker_wpm.groupby("grade", as_index=False).agg(
        avg_wpm=("avg_wpm_by_speaker", "mean"),
        n_speakers=("uid", "nunique")
    )
    avg_wpm_by_grade.to_csv(out_avg_wpm_by_grade, sep="\t", index=False)

    
    err = word_df.merge(speaker_grade, on="uid", how="left").dropna(subset=["grade"]).copy()
    err["is_correct"] = err["error_cat"].astype(str).eq("Correct")

    
    err_table = err.groupby("grade")["is_correct"].value_counts().unstack(fill_value=0)

    
    err_table = err_table.reindex(columns=[True, False], fill_value=0)

    
    err_table = err_table.rename(columns={True: "Correct", False: "NotCorrect"})

    
    if err_table.empty:
        
        err_table = pd.DataFrame(columns=["Correct", "NotCorrect"])
        err_table.index.name = "grade"

    err_table.to_csv(out_err_table, sep="\t")

    
    if err_table.shape[0] >= 2 and err_table.shape[1] == 2 and err_table.to_numpy().sum() > 0:
        chi2, p, dof, _ = chi2_contingency(err_table.values)
    else:
        chi2, p, dof = np.nan, np.nan, np.nan

    pd.DataFrame([{
        "test": "Chi2 Correct vs NotCorrect by grade",
        "chi2": chi2, "p_value": p, "dof": dof
    }]).to_csv(out_err_stats, sep="\t", index=False)

    os.makedirs(figs_dir, exist_ok=True)

    plot_stacked_histogram(
        per_speaker_wpm.rename(columns={"avg_wpm_by_speaker": "wpm"}),
        value_col="wpm",
        grade_col="grade",
        out_path=os.path.join(figs_dir, "stacked_hist_wpm_by_grade.png"),
        bins=20
    )


def wpm_main(processed_dir: Path):
    
    story_lengths_path = Path(os.path.join(processed_dir, "story_lengths.tsv"))
    story_meta_path = Path(os.path.join(processed_dir, "story.xlsx"))
    word_level_path = Path(os.path.join(processed_dir, "word_level_data.csv"))
    figs_dir = Path(os.path.join(processed_dir, "figures_wpm"))

    
    stories_df = pd.read_csv(story_lengths_path, sep="\t")
    stories_df["uid"] = stories_df["uid"].astype(str).str.strip()
    stories_df["audio_id"] = stories_df["audio_id"].astype(str).str.strip()
    meta = pd.read_excel(story_meta_path)
    col_user = "userId (matches the uid in the recording file name)"
    grade_col = None
    for cand in ["grade", "Grade", "GRADE", "class", "Class"]:
        if cand in meta.columns:
            grade_col = cand
            break
    if grade_col is None:
        grade_col = [c for c in meta.columns if c != col_user][0]
    meta = meta[[col_user, grade_col]].rename(columns={col_user: "uid", grade_col: "grade"})

    word_df = read_word_level(word_level_path)

    
    out_avg_wpm_by_grade = Path(os.path.join(processed_dir, "avg_wpm_by_grade.tsv"))
    out_err_table = Path(os.path.join(processed_dir, "error_correct_table.tsv"))
    out_err_stats = Path(os.path.join(processed_dir, "error_correct_stats.tsv"))

    build_wpm_tables(
        word_df=word_df,
        stories_df=stories_df,
        meta_df=meta,
        out_avg_wpm_by_grade=out_avg_wpm_by_grade,
        out_err_table=out_err_table,
        out_err_stats=out_err_stats,
        figs_dir=figs_dir
    )

    print(f"saved avg wpm by grade to: {out_avg_wpm_by_grade}")
    print(f"saved error correct table to: {out_err_table}")
    print(f"saved error correct stats to: {out_err_stats}")
    print(f"saved wpm histogram to: {os.path.join(figs_dir, 'stacked_hist_wpm_by_grade.png')}")

if __name__ == "__main__":
#    ROOT_DIR = os.path.join("..", "..")
#    PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_annotations")
    PROCESSED_DIR = "processed_annotations"
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    
    in_path = Path(os.path.join(PROCESSED_DIR, "utterance_windows_modified.csv"))
    out_story_path = Path(os.path.join(PROCESSED_DIR, "story_lengths.tsv"))
    out_sum_path = Path(os.path.join(PROCESSED_DIR, "story_sum_durations.tsv"))
    story_excel_path = Path(os.path.join(PROCESSED_DIR, "story.xlsx"))
    out_grade_means_path = Path(os.path.join(PROCESSED_DIR, "grade_means.tsv"))
    out_grade_counts_path = Path(os.path.join(PROCESSED_DIR, "grade_counts.tsv"))
    out_stats_path = Path(os.path.join(PROCESSED_DIR, "grade_stats.tsv"))
    figs_dir = Path(os.path.join(PROCESSED_DIR, "figures"))


    total_time_main(in_path, out_story_path, out_sum_path,
                    story_excel_path,
                    out_grade_means_path,
                    out_grade_counts_path,
                    out_stats_path,
                    figs_dir)

    wpm_main(Path(PROCESSED_DIR))

