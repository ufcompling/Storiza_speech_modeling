from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt
from GenerateGroupedAnnotations import generate_combined_cross_annotation_dicts

# ---------------- Utilities ----------------

def _get_annotator_id(ann: Dict[str, Any]) -> str:
    """
    Robustly extract an annotator identifier from a Labels Studio annotation record.
    Falls back to the annotation id if no user field is present.
    """
    for k in ("annotator_id", "completed_by", "created_by", "updated_by", "created_username", "updated_username"):
        if k in ann and ann[k] is not None:
            return ann[k]["email"]
    # Fallbacks commonly seen
    user = ann.get("author") or ann.get("user") or ann.get("owner")
    if user:
        return str(user)
    # Final fallback: annotation id (not ideal, but prevents crashes)
    return f"ann_{ann.get('id', 'unknown')}"

def _extract_intervals_from_annotation(ann: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Given a transformed annotation (where ann['result'] is a list of span dicts
    with 'start' and 'end'), return a list of valid (start, end) floats.
    """
    intervals: List[Tuple[float, float]] = []
    for item in ann.get("result", []):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        try:
            s = float(start)
            e = float(end)
        except Exception:
            continue
        if e > s:
            intervals.append((s, e))
    return intervals

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals and return a normalized list."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def _measure(intervals: List[Tuple[float, float]]) -> float:
    """Total duration covered by intervals (assumes disjoint)."""
    return sum(e - s for s, e in intervals)

def _intersection_measure(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Total overlap length between two (assumed merged) interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        start = max(s1, s2)
        end = min(e1, e2)
        if end > start:
            total += end - start
        # advance the interval which ends first
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return total

def _tiou(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """
    Temporal IoU between two sets of intervals:
    IoU = intersection(a,b) / union(a,b)
    """
    if not a and not b:
        return 1.0
    A = _merge_intervals(a)
    B = _merge_intervals(b)
    inter = _intersection_measure(A, B)
    union = _measure(A) + _measure(B) - inter
    if union <= 0:
        return 0.0
    return inter / union

# ---------------- Core computations ----------------

def compute_pairwise_shared_counts(annotation_dict: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[List[int]]]:
    """
    For a dict: audio -> [annotations], compute how many audios each pair of annotators co-annotated.
    Returns (annotator_ids, matrix) where matrix[i][j] is count of shared audios.
    Diagonal equals the number of audios each annotator annotated within this dict.
    """
    # Collect the full annotator set
    annotators: set = set()
    # For each audio, find set of annotators who have at least one annotation
    audio_to_annos: Dict[str, set] = {}

    for audio, anns in annotation_dict.items():
        ann_ids = set()
        for ann in anns:
            ann_ids.add(_get_annotator_id(ann))
        if ann_ids:
            audio_to_annos[audio] = ann_ids
            annotators.update(ann_ids)

    annotator_ids = sorted(annotators)
    idx = {aid: i for i, aid in enumerate(annotator_ids)}
    n = len(annotator_ids)
    M = [[0 for _ in range(n)] for _ in range(n)]

    for audio, ann_ids in audio_to_annos.items():
        # diagonal: each annotator gets a +1 for this audio they annotated
        for aid in ann_ids:
            i = idx[aid]
            M[i][i] += 1
        # off-diagonal: for each unordered pair, add 1
        ids_list = list(ann_ids)
        for i in range(len(ids_list)):
            for j in range(i + 1, len(ids_list)):
                ii = idx[ids_list[i]]
                jj = idx[ids_list[j]]
                M[ii][jj] += 1
                M[jj][ii] += 1

    return annotator_ids, M

def compute_pairwise_average_tiou(annotation_dict: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[List[Optional[float]]]]:
    """
    For a dict: audio -> [annotations], compute average pairwise temporal IoU per annotator pair.
    Returns (annotator_ids, matrix) where matrix[i][j] is the average IoU over audios both annotated.
    Diagonal is set to 1.0 if the annotator has at least one audio, else None.
    """
    # Build set of annotators and also index annotations per (audio, annotator)
    annotators: set = set()
    # audio -> annotator -> list of intervals
    audio_annot_intervals: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for audio, anns in annotation_dict.items():
        per_annot: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for ann in anns:
            aid = _get_annotator_id(ann)
            annotators.add(aid)
            per_annot[aid].extend(_extract_intervals_from_annotation(ann))
        # Merge to speed up computation later
        for aid in per_annot:
            per_annot[aid] = _merge_intervals(per_annot[aid])
        if per_annot:
            audio_annot_intervals[audio] = per_annot

    annotator_ids = sorted(annotators)
    idx = {aid: i for i, aid in enumerate(annotator_ids)}
    n = len(annotator_ids)
    sum_iou = [[0.0 for _ in range(n)] for _ in range(n)]
    cnt = [[0 for _ in range(n)] for _ in range(n)]

    for audio, per_annot in audio_annot_intervals.items():
        ids = list(per_annot.keys())
        for i in range(len(ids)):
            ai = ids[i]
            Ii = per_annot[ai]
            for j in range(i, len(ids)):
                aj = ids[j]
                Ij = per_annot[aj]
                iou = _tiou(Ii, Ij)
                ii = idx[ai]
                jj = idx[aj]
                sum_iou[ii][jj] += iou
                cnt[ii][jj] += 1
                if ii != jj:
                    sum_iou[jj][ii] += iou
                    cnt[jj][ii] += 1

    # Build final matrix, using None when no shared audios
    M: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if cnt[i][j] > 0:
                M[i][j] = sum_iou[i][j] / cnt[i][j]
            elif i == j:
                M[i][j] = 1.0  # no data, but identity
            else:
                M[i][j] = None

    return annotator_ids, M

def compute_pairwise_mean_abs_duration_diff(
    annotation_dict: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[str], List[List[Optional[float]]], List[List[Optional[float]]], List[List[int]]]:
    """
    For each audio annotated by both annotators i and j:
      - Let Di = total annotated duration by i (sum of merged spans)
      - Let Dj = total annotated duration by j
      - Compute |Di - Dj| for that audio
    Returns (annotator_ids, mean_matrix, sd_matrix, count_matrix) over audios they share.
    """
    from collections import defaultdict

    # audio -> annotator -> merged intervals + durations
    annotators: set = set()
    audio_annot_intervals: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for audio, anns in annotation_dict.items():
        per_annot: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for ann in anns:
            aid = _get_annotator_id(ann)
            annotators.add(aid)
            per_annot[aid].extend(_extract_intervals_from_annotation(ann))
        for aid in per_annot:
            per_annot[aid] = _merge_intervals(per_annot[aid])
        if per_annot:
            audio_annot_intervals[audio] = per_annot

    annotator_ids = sorted(annotators)
    idx = {aid: i for i, aid in enumerate(annotator_ids)}
    n = len(annotator_ids)

    sum_abs = [[0.0 for _ in range(n)] for _ in range(n)]
    sumsq_abs = [[0.0 for _ in range(n)] for _ in range(n)]
    cnt = [[0 for _ in range(n)] for _ in range(n)]

    for _, per_annot in audio_annot_intervals.items():
        ids = list(per_annot.keys())
        # precompute durations for speed
        durations = {aid: sum(e - s for (s, e) in per_annot[aid]) for aid in ids}
        for a_i in range(len(ids)):
            ai = ids[a_i]
            Di = durations[ai]
            for a_j in range(a_i, len(ids)):
                aj = ids[a_j]
                Dj = durations[aj]
                d = abs(Di - Dj)
                ii, jj = idx[ai], idx[aj]
                sum_abs[ii][jj]  += d
                sumsq_abs[ii][jj]+= d * d
                cnt[ii][jj]      += 1
                if ii != jj:
                    sum_abs[jj][ii]   += d
                    sumsq_abs[jj][ii] += d * d
                    cnt[jj][ii]       += 1

    meanM: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]
    sdM:   List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if cnt[i][j] > 0:
                m = sum_abs[i][j] / cnt[i][j]
                meanM[i][j] = m
                if cnt[i][j] > 1:
                    var = max(0.0, (sumsq_abs[i][j] - cnt[i][j]*m*m) / (cnt[i][j]-1))
                    sdM[i][j] = math.sqrt(var)
                else:
                    sdM[i][j] = None
            elif i == j:
                meanM[i][j] = 0.0
                sdM[i][j] = None
            else:
                meanM[i][j] = None
                sdM[i][j] = None

    return annotator_ids, meanM, sdM, cnt

# ---------------- Plotting ----------------

def plot_triangular_heatmap(
    matrix: List[List[Optional[float]]],
    labels: List[str],
    title: str,
    null_label: str = "NA",
    value_format: str = "{:.2f}",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Render a bottom-left triangular heatmap (strictly lower triangle; diagonal excluded).
    The top-right triangle and diagonal are blacked out.
    Any cell matching the null value is rendered with a blank text label.
    - matrix: 2D list of floats or None (or 'NA' strings)
    - labels: annotator labels for both axes
    - title: figure title
    - null_label: if matrix entries are strings, treat this as the "null" marker
    """
    n = len(matrix)
    arr = np.empty((n, n), dtype=float)
    arr[:] = np.nan

    # Fill numeric entries; treat None or 'null_label' strings as NaN
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if isinstance(v, (int, float)):
                arr[i, j] = float(v)
            elif isinstance(v, str) and v == null_label:
                arr[i, j] = np.nan
            else:
                if v is None:
                    arr[i, j] = np.nan
                else:
                    try:
                        arr[i, j] = float(v)
                    except Exception:
                        arr[i, j] = np.nan

    # Mask the upper triangle AND the diagonal (keep strictly lower triangle only)
    # The original code already does this by setting values to np.nan for j >= i
    # We will use this to draw the bottom-left triangle.

    # Create a new array for plotting the "blacked out" effect
    plot_data = np.copy(arr)
    for i in range(n):
        for j in range(n):
            if j >= i:  # above or on diagonal
                plot_data[i, j] = -1  # A distinct value to represent "blacked out" in the colormap

    # Use a masked array for text annotations, so NaNs are not rendered at all
    text_data = np.ma.masked_invalid(arr)

    # Make the figure larger
    fig, ax = plt.subplots(figsize=(10, 8)) # Increased figure size

    # Define a colormap for the heatmap.
    # We want the values in the lower triangle to have a color gradient,
    # and the upper triangle/diagonal to be black.
    # The default 'viridis' colormap goes from yellow (high) to purple (low).
    # We will ensure -1 maps to black.
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # NaNs (unfilled cells in the lower triangle) will be white
    cmap.set_under(color='black') # Values less than vmin will be black.
                                  # We'll set vmin to 0 so -1 becomes black.

    # Determine the range for the actual data (excluding -1 and NaNs)
    valid_data = arr[~np.isnan(arr)]
    vmin = np.min(valid_data) if valid_data.size > 0 else 0
    vmax = np.max(valid_data) if valid_data.size > 0 else 1

    im = ax.imshow(plot_data, interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=vmax) # Adjusted vmin to ensure -1 is below the threshold

    ax.set_title(title, fontsize=16) # Increased title font size
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    labels = [l.split("@")[0] for l in labels]  # quick way to get names
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12) # Increased label font size and rotation for better readability
    ax.set_yticklabels(labels, fontsize=12) # Increased label font size

    # Add text annotations for strictly lower triangle only (exclude diagonal)
    for i in range(n):
        for j in range(n):
            if j >= i:  # skip diagonal and upper triangle
                continue
            v = matrix[i][j]
            is_null = (v is None) or (isinstance(v, str) and v == null_label)
            if not is_null:
                try:
                    text = value_format.format(float(v))
                except Exception:
                    text = ""
            else:
                text = ""
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10) # Increased text font size

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300) # Increased DPI for higher resolution
    if show:
        plt.show()
    return fig, ax


def calculate_matrix_average(M: List[List[Optional[float]]]) -> Optional[float]:
    """
    Average of off-diagonal entries only (ignores None/NA and the diagonal).
    Returns None if there are no valid off-diagonal entries.
    """
    total = 0.0
    count = 0
    n = len(M)
    for i, row in enumerate(M):
        for j, item in enumerate(row):
            if i == j:     # skip diagonal
                continue
            if item is not None:
                total += item
                count += 1
    return (total / count) if count else None


# ---------------- Demo / CLI for v1_v2_overlap ----------------

if __name__ == "__main__":
    # Generate dicts and focus on v1_v2_overlap (as requested)
    dicts = generate_combined_cross_annotation_dicts()
    overlap = dicts["all_cross_overlap"]

    # 1) Pairwise shared counts
    ids_shared, M_shared = compute_pairwise_shared_counts(overlap)
    plot_triangular_heatmap(
        M_shared,
        ids_shared,
        title="Shared Audio Counts (all_cross_overlap)",
        null_label="NA",
        value_format="{:.0f}",
        save_path=None,
        show=True,
    )

    print("Aveage shared_count:", calculate_matrix_average(M_shared))

    # 2) Pairwise average tIoU
    ids_tiou, M_tiou = compute_pairwise_average_tiou(overlap)

    print("Aveage tiou:", calculate_matrix_average(M_tiou))

    plot_triangular_heatmap(
        M_tiou,
        ids_tiou,
        title="Average Temporal IoU (all_cross_overlap)",
        null_label="NA",
        value_format="{:.3f}",
        save_path=None,
        show=True,
    )

