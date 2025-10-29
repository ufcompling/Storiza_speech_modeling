# ==== α_u / cu-α wrappers using your KrippendorffUnitizedAlpha classes ====
import json
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations
import numpy as np
from tqdm import tqdm


from krippendorff_unitized_alpha import (
    Segment as KUSegment,
    KrippendorffUnitizedAlpha,
    impute_empty_segments,
    EMPTY_TAG,
)

from GenerateGroupedAnnotations import generate_combined_cross_annotation_dicts
from Labels import TOP_LEVEL_LABELS, ORTHO_SPEC, GRAM_SPEC, STRUCT_SPEC, VISUAL_SPEC, RUNON_SPEC, FULL_COMBOS
from calculate_alignment_scores import plot_triangular_heatmap


# Reuse your extractor for LS-style blocks or value fields
def _extract_category_token_for_alpha(
    label: Dict[str, Any],
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
) -> str:
    block = label.get(subDictName) if subDictName else label

    # 1) Value-field mode
    if len(target_values) == 1 and isinstance(target_values[0], str):
        key = target_values[0]
        if isinstance(block, dict) and key in block:
            val = block.get(key, None)
            sval = ("" if val is None else str(val).strip())
            return sval if sval else blank_label

    # 2) Multi-label (checkbox) mode
    if isinstance(block, dict):
        if "labels" in block and "selected" in block:
            sel_map = dict(zip(block["labels"], block["selected"]))
        else:
            sel_map = block
        active = [name for name in target_values if bool(sel_map.get(name, False))]
        if not active:
            return blank_label
        return "+".join(sorted(active))

    return blank_label


def _to_kua_segments_for_audio(
    annotation_data: list,
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
    time_scale: int = 1000,  # quantize seconds -> milliseconds by default
    keep_blank: bool = False, # for cu-α keep only real units; α_u will add EMPTY via imputation
) -> Tuple[Dict[str, List[KUSegment]], int]:
    """
    Build { coder : [KUSegment(...)] } and return with max_length (quantized).
    Ensures non-overlap by rounding starts/ends and nudging zero-lengths.
    """
    data: Dict[str, List[KUSegment]] = {}
    max_end = 0

    def q(t: Optional[float]) -> int:
        if t is None:
            return 0
        return int(round(float(t) * time_scale))

    for ann in annotation_data:
        coder = ann["completed_by"]["email"].split("@")[0]
        lst = data.setdefault(coder, [])
        for item in ann.get("result", []):
            start = q(item.get("start"))
            end = q(item.get("end"))
            if end <= start:
                # nudge a degenerate segment to length 1
                end = start + 1
            cat = _extract_category_token_for_alpha(item, target_values, subDictName, blank_label)
            if (not keep_blank) and (cat == blank_label):
                # For α_u we will impute EMPTY later; for cu-α we ignore blanks entirely
                pass
            else:
                lst.append(KUSegment(tag=cat, start=start, end=end))
                if end > max_end:
                    max_end = end

    # Sort and fix tiny overlaps due to rounding; also merge adjacent same-tag to reduce fragmentation
    for coder, segs in data.items():
        segs.sort(key=lambda s: (s.start, s.end))
        merged: List[KUSegment] = []
        for s in segs:
            if not merged:
                # ensure positive length
                if s.end <= s.start:
                    s = KUSegment(tag=s.tag, start=s.start, end=s.start + 1)
                merged.append(s)
                continue

            prev = merged[-1]

            # 1) Overlap or touch? push forward to be at least prev.end + 1
            if s.start <= prev.end:
                s = KUSegment(tag=s.tag, start=prev.end + 1, end=max(s.end, prev.end + 2))

            # 2) If there is exactly a 1-unit gap (s.start == prev.end + 2),
            #    snap to adjacency to avoid zero-length EMPTY in the imputer.
            if s.start == prev.end + 2:
                s = KUSegment(tag=s.tag, start=prev.end + 1, end=max(s.end, prev.end + 2))

            # 3) Guarantee positive length
            if s.end <= s.start:
                s = KUSegment(tag=s.tag, start=s.start, end=s.start + 1)

            # 4) Merge if now adjacent and same tag
            if s.tag == prev.tag and s.start == prev.end + 1:
                merged[-1] = KUSegment(tag=prev.tag, start=prev.start, end=s.end)
            else:
                merged.append(s)

        data[coder] = merged

    return data, max_end


def _alpha_u_single_audio(
    annotation_data: list,
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
    time_scale: int = 1000,
) -> float:
    # Build only “real” units; blanks will be added as EMPTY
    data, max_len = _to_kua_segments_for_audio(
        annotation_data, target_values, subDictName, blank_label, time_scale, keep_blank=False
    )
    if len(data) < 2:
        return float("nan")
    items=[]
    for val in data.values():
        items.extend(val)
    if not any(items):
        return float("nan")
    # Impute EMPTY to cover the whole continuum (α_u requirement)
    data_imputed = {
        coder: impute_empty_segments(segs, max_len) for coder, segs in data.items()
    }
    try:
        return float(KrippendorffUnitizedAlpha(data_imputed).result)
    except Exception:
        return float("nan")


def _cu_alpha_single_audio(
    annotation_data: list,
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
    time_scale: int = 1000,
) -> float:
    # Build only “real” units and DO NOT impute empties (cu-α ignores missing)
    data, _ = _to_kua_segments_for_audio(
        annotation_data, target_values, subDictName, blank_label, time_scale, keep_blank=False
    )
    if len(data) < 2:
        return float("nan")
    try:
        return float(KrippendorffUnitizedAlpha(data).result)
    except Exception:
        return float("nan")


def _nanmean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def compute_alpha_u_or_cu_stats_for_dataset(
    data_dict: Dict[str, list],
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
    time_scale: int = 1000,
) -> Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]:
    """
    Returns:
      overall_mean, overall_sd, pairwise_mean_matrix, pairwise_sd_matrix, annotator_order
    Lower triangle is filled (compatible with your triangular heatmap).
    """
    # annotator universe
    annotators = sorted({
        ann["completed_by"]["email"].split("@")[0]
        for _, ann_list in data_dict.items()
        for ann in ann_list
    })
    idx = {name: i for i, name in enumerate(annotators)}
    N = len(annotators)

    overall_vals: List[float] = []
    pair_accum: Dict[Tuple[int, int], List[float]] = {(i, j): [] for i in range(N) for j in range(N) if i > j}

    # Per-audio computations
    for _, audio_annotations in data_dict.items():
        # overall (all present)
        coders_present = sorted({ann["completed_by"]["email"].split("@")[0] for ann in audio_annotations})
        if len(coders_present) >= 2:
            val = _alpha_u_single_audio(audio_annotations, target_values, subDictName, blank_label, time_scale)

            if not np.isnan(val):
                overall_vals.append(val)

        # pairwise
        for a, b in combinations(coders_present, 2):
            sub = [ann for ann in audio_annotations
                   if ann["completed_by"]["email"].split("@")[0] in (a, b)]
            g = _alpha_u_single_audio(sub, target_values, subDictName, blank_label, time_scale)

            if not np.isnan(g):
                i, j = idx[a], idx[b]
                if i < j:  # fill lower triangle
                    i, j = j, i
                pair_accum[(i, j)].append(g)

    overall_mean, overall_sd = _nanmean_std(overall_vals)

    # Build lower-triangular mean/sd matrices
    mean_mat: List[List[Optional[float]]] = [[None for _ in range(N)] for _ in range(N)]
    sd_mat:   List[List[Optional[float]]] = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j or j > i:
                continue
            vals = pair_accum.get((i, j), [])
            m, s = _nanmean_std(vals)
            mean_mat[i][j] = m if not np.isnan(m) else None
            sd_mat[i][j]   = s if not np.isnan(s) else None

    return overall_mean, overall_sd, mean_mat, sd_mat, annotators


# Convenience runners mirroring your γ “combinations”
def compute_all_alpha_u_and_cu_summaries(
    data_dict: Dict[str, list],
) -> Dict[str, Dict[str, Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]]]:
    """
    Returns:
      {
        "<name>": {
           "alpha_u": (overall_m, overall_s, pair_m, pair_s, labels),
           "cu_alpha": (overall_m, overall_s, pair_m, pair_s, labels),
        },
        ...
      }
    """


    out: Dict[str, Dict[str, Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]]] = {}
    means=[]
    for name, target_values, subname in tqdm(FULL_COMBOS):
        overall_mean, overall_sd, mean_mat, sd_mat, annotators=compute_alpha_u_or_cu_stats_for_dataset(
            data_dict, target_values, subname
        )
        means.append((name,overall_mean))
        out[name] = {
            "alpha_u": (overall_mean, overall_sd, mean_mat, sd_mat, annotators),
        }
    for mean in means:
        print(mean[0]+":",mean[1])
    return out

if __name__ =="__main__":
    dicts = generate_combined_cross_annotation_dicts()
    data = dicts["all_cross_overlap"]

    # Or one setting + heatmaps
    overall_m, overall_s, pair_m, pair_s, labels = compute_alpha_u_or_cu_stats_for_dataset(
        data, TOP_LEVEL_LABELS, "general_label_type"
    )
    print(overall_m,overall_s)

    plot_triangular_heatmap(pair_m, labels, "Pairwise αᵤ (mean)")
    plot_triangular_heatmap(pair_s, labels, "Pairwise αᵤ (sd)")


    # Run all combos (αᵤ + cu-α), save JSON like you do for γ
    all_alpha = compute_all_alpha_u_and_cu_summaries(data)

    json.dump(all_alpha, open("alpha_summary.json", "w"))

