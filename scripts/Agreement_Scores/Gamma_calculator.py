import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Iterable, FrozenSet
from itertools import combinations

import numpy as np
from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity
from pyannote.core import Segment
from sortedcontainers import SortedSet
from tqdm import tqdm

from GenerateGroupedAnnotations import generate_combined_cross_annotation_dicts
from Labels import *
from scripts.Agreement_Scores.calculate_alignment_scores import plot_triangular_heatmap


# -------------------------------------------------------------------
# Canonical labeling helpers
# -------------------------------------------------------------------
def get_annotator_name(annotator_data: Dict[str, Any]) -> str:
    return annotator_data["completed_by"]["email"].split("@")[0]


def _canonical_combo_from_block(
        block: Dict[str, Any],
        target_values: List[str],
        blank: str = "__NONE__",
) -> str:
    """
    Build a canonical category token that can mix:
      • Checkbox-style labels (True/False)
      • Value-bearing fields: intended_word, produced_word, mispronunciation_ipa

    Examples:
      - active checkboxes: "Correct+Orthographic Error"
      - values present:    "produced_word=cat+mispronunciation_ipa=kæt"
      - mixed:             "Correct+produced_word=cat"

    Returns blank if nothing contributes.
    """
    if not isinstance(block, dict):
        return blank

    # Normalize checkbox map if LS-style {"labels":[...], "selected":[...]}
    if "labels" in block and "selected" in block:
        sel_map = dict(zip(block.get("labels", []), block.get("selected", [])))
    else:
        sel_map = block

    special_value_keys = {"intended_word", "produced_word", "mispronunciation_ipa"}

    # 1) Checkbox-style parts (exclude special value keys)
    checkbox_parts = [
        name for name in target_values
        if name not in special_value_keys and bool(sel_map.get(name, False))
    ]
    checkbox_parts.sort()  # canonical order

    # 2) Value-bearing parts (emit as key=value, only if non-empty)
    value_parts = []
    for key in sorted(k for k in target_values if k in special_value_keys):
        if key in block:
            val = block.get(key, None)
            sval = "" if val is None else str(val).strip()
            if sval:
                value_parts.append(f"{key}={sval}")

    parts = checkbox_parts + value_parts
    if not parts:
        return blank
    return "+".join(parts)




def generate_audio_continuua(
        annotation_data: list,
        target_values: List[str],
        subDictName: Optional[str] = None,
        blank_label: str = "__NONE__"
) -> Dict[str, Continuum]:
    """
    Build { annotator_email : Continuum } for ONE audio.
    Category is a canonical token of *active* top-levels, not the whole T/F vector.
    """
    continuua: Dict[str, Continuum] = {}

    for annotator in annotation_data:
        annotator_name = get_annotator_name(annotator)
        c = continuua.setdefault(annotator_name, Continuum())

        for label in annotator["result"]:
            start = label.get("start")
            end = label.get("end")
            if start is None or end is None or end <= start:
                continue

            block = label.get(subDictName) if subDictName else label
            cat_token = _canonical_combo_from_block(block, target_values, blank=blank_label)

            # (Optional) debug print
            # print(annotator_name, Segment(start, end), cat_token)
            if cat_token != blank_label:
                c.add(annotator_name, Segment(start, end), cat_token)

    return continuua


def _continuum_nonempty(cont: Continuum) -> bool:
    try:
        return len(cont.annotators) > 0
    except Exception:
        # Fail-open: if API changes, let gamma call decide
        return True


def _merge_many(continua: List[Continuum]) -> Continuum:
    if not continua:
        return Continuum()
    merged = continua[0]
    for c in continua[1:]:
        merged = merged.merge(c, in_place=False)

    for c in continua:
        if len(c.annotators):
            merged.add_annotator(c.annotators[0])
    return merged


def calculate_pairwise_agreement(
        continua_by_annotator: Dict[str, Continuum],
        alpha: float = 1.0,
        beta: float = 1.0,
) -> Dict[FrozenSet[str], float]:
    """
    Pairwise γ for every unordered annotator pair.

    Returns:
      { frozenset({ann_i, ann_j}): gamma_value }
    """
    dissim = CombinedCategoricalDissimilarity(alpha=alpha, beta=beta)

    names = sorted(a for a, c in continua_by_annotator.items() if True)
    out: Dict[FrozenSet[str], float] = {}

    for a, b in combinations(names, 2):
        pair = continua_by_annotator[a].merge(continua_by_annotator[b], in_place=False)

        non_empty = [c for c in (continua_by_annotator[a],continua_by_annotator[b]) if _continuum_nonempty(c)]
        if len(non_empty) < 1:
            continue

        pair._categories = SortedSet(list(pair.category_weights.keys()))
        g = pair.compute_gamma(dissim, n_samples=100, ).gamma
        out[frozenset({a, b})] = g

    return out


def calculate_overall_gamma(
        continua_by_annotator: Dict[str, Continuum],
        alpha: float = 1.0,
        beta: float = 1.0,
        show_image: bool = False,
) -> float:
    """
    γ across ALL annotators for a single audio (merge everything once).

    Returns:
      gamma (float). If <2 non-empty annotators, returns NaN.
    """
    non_empty = [c for c in continua_by_annotator.values() if _continuum_nonempty(c)]
    if len(non_empty) < 1:
        return float("nan")

    merged = _merge_many([continua_by_annotator[name] for name in sorted(continua_by_annotator)])

    if show_image:
        try:
            from pygamma_agreement import show_continuum
            show_continuum(merged, labelled=True)
        except Exception:
            pass

    dissim = CombinedCategoricalDissimilarity(alpha=alpha, beta=beta)
    merged._categories = SortedSet(list(merged.category_weights.keys()))
    if not merged.num_units:
        return float("nan")
    return merged.compute_gamma(dissim, n_samples=100).gamma
def _extract_category_token(
    label: Dict[str, Any],
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__"
) -> str:
    """
    If target_values looks like an LS-style inventory (multiple values),
      -> build canonical "+".joined token of active ones (subset of target_values).
    If target_values has exactly one entry and that key exists in the (sub)block,
      -> treat the *value* at that key as the category (stringified).
    Otherwise -> blank_label.
    """
    block = label.get(subDictName) if subDictName else label

    # Value-field mode: single field name whose *value* is the category
    if len(target_values) == 1 and isinstance(target_values[0], str):
        key = target_values[0]
        if isinstance(block, dict) and key in block:
            val = block.get(key, None)
            if val is None:
                return blank_label
            sval = str(val).strip()
            return sval if sval else blank_label

    # Multi-label (checkbox) mode
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


def _build_continua_for_audio(
    annotation_data: list,
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
) -> Dict[str, Continuum]:
    """Build { annotator_name : Continuum } for one audio using _extract_category_token."""
    out: Dict[str, Continuum] = {}
    for annotator in annotation_data:
        annotator_name = annotator["completed_by"]["email"].split("@")[0]
        c = out.setdefault(annotator_name, Continuum())

        for label in annotator.get("result", []):
            start = label.get("start")
            end = label.get("end")
            if start is None or end is None or end <= start:
                continue

            cat = _extract_category_token(label, target_values, subDictName, blank_label)
            if cat !=blank_label:
                c.add(annotator_name, Segment(start, end), cat)
    return out


def _combine_to_single_continuum(continua_by_annotator: Dict[str, Continuum]) -> Continuum:
    """
    Safer than chaining Continuum.merge: create a fresh Continuum and re-add all units.
    This guarantees the combined continuum's category universe matches what's inside.
    """
    combined = Continuum()
    for name in sorted(continua_by_annotator):
        combined.add_annotator(name)
        for _, unit in continua_by_annotator[name]:
            combined.add(name, unit.segment, unit.annotation)
    for name in sorted(continua_by_annotator):
        combined.add_annotator(name)
    return combined


def _compute_gamma(
    cont: Continuum,
    alpha: float = 1.0,
    beta: float = 1.0,
    n_samples: int = 60,
    fast: bool = False,
) -> float:
    if len(cont.annotators) < 2:
        return float("nan")
    dissim = CombinedCategoricalDissimilarity(alpha=alpha, beta=beta)
    # Use kwargs that pygamma-agreement exposes; fast is supported in modern versions.
    try:
        return cont.compute_gamma(dissim, n_samples=n_samples, fast=fast).gamma
    except TypeError:
        # Fallback if `fast` not supported in  installed version
        return cont.compute_gamma(dissim, n_samples=n_samples).gamma


def _nanmean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))

def compute_gamma_stats_for_dataset(
    data_dict: Dict[str, list],
    target_values: List[str],
    subDictName: Optional[str],
    blank_label: str = "__NONE__",
    alpha: float = 1.0,
    beta: float = 1.0,
    n_samples: int = 60,
) -> Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]:
    """
    data_dict: mapping audio_id -> list[annotator_json] ( generate_combined_cross_annotation_dicts() style)
    Returns:
      overall_mean, overall_sd,
      pairwise_mean_matrix (NxN, lower triangle used),
      pairwise_sd_matrix (NxN, lower triangle used),
      annotator_order (list[str])  # pass this to  triangular heatmap
    """
    # 1) Collect the universe of annotators across the dataset
    annotators_set: set[str] = set()
    for _, ann_list in data_dict.items():
        for ann in ann_list:
            annotators_set.add(ann["completed_by"]["email"].split("@")[0])
    annotator_order = sorted(annotators_set)
    idx = {name: i for i, name in enumerate(annotator_order)}
    N = len(annotator_order)

    # 2) Prepare accumulators
    overall_vals: List[float] = []
    pair_accum: Dict[Tuple[int, int], List[float]] = {(i, j): [] for i in range(N) for j in range(N) if i > j}

    # 3) Iterate every audio
    for _, audio_annotations in tqdm(data_dict.items()):
        # build per-audio continua
        continua = _build_continua_for_audio(audio_annotations, target_values, subDictName, blank_label)

        # overall gamma (at least one annotator present *for this audio*)
        if len(continua) >= 1:
            combined = _combine_to_single_continuum(continua)
            if not combined.num_units:
                continue
            g_all = _compute_gamma(combined, alpha=alpha, beta=beta, n_samples=n_samples, fast=False)
            if not np.isnan(g_all):
                overall_vals.append(g_all)

        # pairwise gammas for annotators who appear on this audio
        present = sorted(continua.keys())
        for a, b in combinations(present, 2):
            # combine only those two
            sub = {a: continua[a], b: continua[b]}
            two = _combine_to_single_continuum(sub)
            if not two.num_units:
                continue
            g = _compute_gamma(two, alpha=alpha, beta=beta, n_samples=n_samples, fast=False)
            if not np.isnan(g):
                i, j = idx[a], idx[b]
                if i < j:
                    i, j = j, i
                pair_accum[(i, j)].append(g)

    # 4) Compute overall mean/sd
    overall_mean, overall_sd = _nanmean_std(overall_vals)

    # 5) Build lower-triangular matrices of mean/sd
    mean_mat: List[List[Optional[float]]] = [[None for _ in range(N)] for _ in range(N)]
    sd_mat:   List[List[Optional[float]]] = [[None for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == j or j > i:
                # keep None ->  triangular heatmap blanks these
                continue
            vals = pair_accum.get((i, j), [])
            m, s = _nanmean_std(vals)
            mean_mat[i][j] = m if not np.isnan(m) else None
            sd_mat[i][j]   = s if not np.isnan(s) else None
    print("overall_mean:", overall_mean,"overall_sd:", overall_sd)
    return overall_mean, overall_sd, mean_mat, sd_mat, annotator_order

def _compute_one_combo(args):
    """Helper so it’s picklable by multiprocessing."""
    name, target_values, subname, data_dict = args
    res = compute_gamma_stats_for_dataset(
        data_dict=data_dict,
        target_values=target_values,
        subDictName=subname,
    )
    return name, res


def _compute_one_combo_with_audio(args):
    """
    Returns:
      (combo_name,
       summary_tuple_for_json,   # same as before from compute_gamma_stats_for_dataset
       per_audio_gamma_dict)     # { audio_id: {"annotators": [...], combo_name: gamma_or_nan}, ... }
    """
    (name, target_values, subname, data_dict, alpha, beta, n_samples) = args

    # (A) Summary for JSON (overall mean/sd + pairwise matrices)
    summary = compute_gamma_stats_for_dataset(
        data_dict=data_dict,
        target_values=target_values,
        subDictName=subname,
        alpha=alpha,
        beta=beta,
        n_samples=n_samples,
    )

    # (B) Per-audio γ for this combo
    per_audio: Dict[str, Dict[str, Any]] = {}
    dissim = CombinedCategoricalDissimilarity(alpha=alpha, beta=beta)

    for audio_id, audio_annotations in data_dict.items():
        # all annotators present on this audio (names only, even if no units for this combo)
        annotators = sorted({ann["completed_by"]["email"].split("@")[0] for ann in audio_annotations})

        # build continua for this combo
        continua = _build_continua_for_audio(
            audio_annotations, target_values, subDictName=subname, blank_label="__NONE__"
        )

        # Combine into a single Continuum and register all annotators so they are retained
        combined = Continuum()
        for a in annotators:
            combined.add_annotator(a)
        for a_name, cont in continua.items():
            for _, unit in cont:
                combined.add(a_name, unit.segment, unit.annotation)

        # compute γ for this audio/combo
        if combined.num_units < 1 or len(combined.annotators) < 2:
            gamma_val = float("nan")
        else:
            try:
                gamma_val = combined.compute_gamma(dissim, n_samples=n_samples).gamma
            except TypeError:
                gamma_val = combined.compute_gamma(dissim, n_samples=n_samples).gamma

        per_audio[audio_id] = {"annotators": annotators, name: float(gamma_val)}

    return name, summary, per_audio


def compute_all_gamma_summaries(
    data_dict: Dict[str, list],
    max_workers: int = 16,
    alpha: float = 1.0,
    beta: float = 1.0,
    n_samples: int = 60,
) -> Tuple[
    Dict[str, Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]],  # json_out
    Dict[str, Dict[str, Any]]  # audio_out: {audio_id: {"annotators":[...], "<combo1>":γ, "<combo2>":γ, ...}}
]:
    """
    Returns:
      json_out  : same structure you already save to JSON (per-combo summaries)
      audio_out : per-audio rows you can write to TSV later
    """
    combos = FULL_COMBOS
    num_workers = min(max_workers, os.cpu_count() or 1)

    json_out = {}
    audio_out: Dict[str, Dict[str, Any]] = {}

    tasks = [
        (name, target_values, subname, data_dict, alpha, beta, n_samples)
        for (name, target_values, subname) in combos
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_compute_one_combo_with_audio, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing γ summaries (parallel)"):
            combo_name = futures[fut]
            try:
                name, summary_tuple, per_audio = fut.result()
            except Exception as e:
                print(f"{combo_name} failed: {e}")
                continue

            # (1) JSON output per combo
            json_out[name] = summary_tuple

            # (2) Merge per-audio columns across combos
            for audio_id, row in per_audio.items():
                base = audio_out.setdefault(audio_id, {"annotators": row["annotators"]})
                # keep first annotator list encountered
                if "annotators" not in base or not base["annotators"]:
                    base["annotators"] = row["annotators"]
                base[name] = row[name]

    return json_out, audio_out



import math

def save_audiowise_tsv(audio_out: Dict[str, Dict[str, Any]], path: str,
                       combos_order: Optional[List[str]] = None):
    """
    Write TSV with columns:
      audio_id, annotators, worst_score, <combo1>, <combo2>, ...

    - worst_score is the minimum across the listed combo columns on that row
    - rows are sorted by worst_score ascending (NaN rows go last)
    """
    if combos_order is None:
        combos_order = [name for (name, _, _) in FULL_COMBOS]

    header = ["audio_id", "annotators", "worst_score"] + combos_order

    # Build rows with a numeric worst_score we can sort by
    rows = []
    for audio_id in audio_out:
        row = audio_out[audio_id]
        annotators = ",".join(row.get("annotators", []))

        # Gather numeric values for combos (ignore None/NaN for worst_score)
        values_numeric = []
        values_str = []
        for name in combos_order:
            v = row.get(name, float("nan"))
            # collect for worst_score
            if isinstance(v, (int, float)) and not math.isnan(v):
                values_numeric.append(float(v))
            # prepare string form for output (match your earlier format)
            if v is None:
                values_str.append("")
            elif isinstance(v, (int, float)):
                values_str.append("NaN" if math.isnan(v) else f"{float(v):.6f}")
            else:
                # fallback if something odd sneaks in
                values_str.append(str(v))

        worst = min(values_numeric) if values_numeric else float("nan")
        worst_str = "NaN" if math.isnan(worst) else f"{worst:.6f}"

        rows.append({
            "audio_id": audio_id,
            "annotators": annotators,
            "worst_score_num": worst,   # for sorting
            "worst_score_str": worst_str,
            "combo_vals_str": values_str
        })

    # Sort: non-NaN first by worst_score asc, NaN last; tie-breaker by audio_id
    def sort_key(r):
        w = r["worst_score_num"]
        isn = math.isnan(w)
        return (1 if isn else 0, w if not isn else float("inf"), r["audio_id"])

    rows.sort(key=sort_key)

    # Write TSV
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(
                [r["audio_id"], r["annotators"], r["worst_score_str"]] + r["combo_vals_str"]
            ) + "\n")



if __name__ == "__main__":



    dicts = generate_combined_cross_annotation_dicts()
    data = dicts["all_cross_overlap"]
    
    sample_link='https://2025storiza.michaelbennie.org/audio_clips/55.0_end_67.2_uid_eyem3EvRZrZsMzdS2HOEHLIsGTs1_sid_Gi6CTd0h6i7rPtveJk1O_1742516188.mp3'
    
    data_simplified = {
        sample_link:
            data[
                sample_link]}


    # Grab one audio’s annotation list
    audio_annotations = data[sample_link]

    # Build per-annotator Continuum with canonical top-level combos
    continua_dict = generate_audio_continuua(
        audio_annotations,
        target_values= ["mispronunciation_ipa"],
        subDictName=None,
        blank_label="__NONE__"
    )

    # 2) Overall γ across all annotators
    overall = calculate_overall_gamma(continua_dict, alpha=1.0, beta=1.0, show_image=True)
    print(f"Overall gamma (all annotators): {overall:.3f}")


    # 1) Pairwise γ
    pairwise = calculate_pairwise_agreement(continua_dict.copy(), alpha=1.0, beta=1.0)
    for pair, g in sorted(pairwise.items(), key=lambda x: tuple(sorted(x[0]))):
        print(f"{sorted(list(pair))}: gamma={g:.3f}")





    overall_m, overall_s, pair_m, pair_s, labels = compute_gamma_stats_for_dataset(
        data_dict=data,
        target_values= ["mispronunciation_ipa"],
        subDictName=None,
    )

    # Heatmap
    plot_triangular_heatmap(pair_m, labels, "Pairwise γ (mean)")
    plot_triangular_heatmap(pair_s, labels, "Pairwise γ (sd)")



    json_out, audio_out = compute_all_gamma_summaries(data, max_workers=1)

    # save JSON (annotator-wise summaries per combo)
    with open("all_gamma_summaries.json", "w") as jf:
        json.dump(json_out, jf)

    # save TSV (per-audio γ per combo)
    save_audiowise_tsv(audio_out, "gamma_per_audio.tsv")

    print("Data Saved")















