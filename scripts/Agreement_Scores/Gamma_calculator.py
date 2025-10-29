import json
from typing import Dict, List, Any, Tuple, Optional, Iterable, FrozenSet
from itertools import combinations

import numpy as np
from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity
from pyannote.core import Segment
from sortedcontainers import SortedSet
from tqdm import tqdm

from GenerateGroupedAnnotations import generate_combined_cross_annotation_dicts
from Labels import TOP_LEVEL_LABELS
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
    Turn a Labels Studio-like block into a canonical category token.

    Accepts either:
      - {"labels":[...], "selected":[bool,...]}   (typical LS sub-dict), or
      - {label_name: bool, ...}                   (already a map)

    Returns a string like "Correct+Orthographic Error" or "__NONE__".
    """
    if "labels" in block and "selected" in block:
        sel_map = dict(zip(block["labels"], block["selected"]))
    else:
        sel_map = block

    active = [name for name in target_values if sel_map.get(name, False)]
    if not active:
        return blank
    # Sorted order to guarantee identical tokens across annotators/audios
    return "+".join(sorted(active))




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

    names = sorted(a for a, c in continua_by_annotator.items() if _continuum_nonempty(c))
    out: Dict[FrozenSet[str], float] = {}

    for a, b in combinations(names, 2):
        pair = continua_by_annotator[a].merge(continua_by_annotator[b], in_place=False)
        pair._categories = SortedSet(list(pair.category_weights.keys()))
        g = pair.compute_gamma(dissim, n_samples=60, ).gamma
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
    if len(non_empty) < 2:
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
    return merged.compute_gamma(dissim, n_samples=60).gamma
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

        # overall gamma (all annotators present *for this audio*)
        if len(continua) >= 2:
            combined = _combine_to_single_continuum(continua)
            g_all = _compute_gamma(combined, alpha=alpha, beta=beta, n_samples=n_samples, fast=False)
            if not np.isnan(g_all):
                overall_vals.append(g_all)

        # pairwise gammas for annotators who appear on this audio
        present = sorted(continua.keys())
        for a, b in combinations(present, 2):
            # combine only those two
            sub = {a: continua[a], b: continua[b]}
            two = _combine_to_single_continuum(sub)
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

def compute_all_gamma_summaries(
    data_dict: Dict[str, list],
) -> Dict[str, Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]]:
    """
    Returns a dict keyed by a readable name, each value is the same 5-tuple that
    compute_gamma_stats_for_dataset returns.
    """
    from Labels import (
        TOP_LEVEL_LABELS,
        PHONO_SPEC,
        DISFLUENCY_SPEC,
        SPECIFIC_LABELS,
    )

    combos = [
        ("Top-level (general_label_type)", TOP_LEVEL_LABELS, "general_label_type"),
        ("Phonological specifics", PHONO_SPEC, "specific_label_type"),
        ("Disfluency specifics", DISFLUENCY_SPEC, "specific_label_type"),
        ("All specifics", SPECIFIC_LABELS, "specific_label_type"),
        ("Intended word", ["intended_word"], None),
        ("Produced word", ["produced_word"], None),
        ("Mispronunciation IPA", ["mispronunciation_ipa"], None),
    ]

    out: Dict[str, Tuple[float, float, List[List[Optional[float]]], List[List[Optional[float]]], List[str]]] = {}
    for name, target_values, subname in tqdm(combos):
        out[name] = compute_gamma_stats_for_dataset(
            data_dict=data_dict,
            target_values=target_values,
            subDictName=subname,
        )
    return out


if __name__ == "__main__":



    dicts = generate_combined_cross_annotation_dicts()
    data = dicts["all_cross_overlap"]




    # Grab one audio’s annotation list
    audio_annotations = next(iter(data.values()))

    # Build per-annotator Continuum with canonical top-level combos
    continua_dict = generate_audio_continuua(
        audio_annotations,
        target_values=TOP_LEVEL_LABELS,
        subDictName="general_label_type",
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
        target_values=TOP_LEVEL_LABELS,
        subDictName="general_label_type",
    )

    # Heatmap
    plot_triangular_heatmap(pair_m, labels, "Pairwise γ (mean)")
    plot_triangular_heatmap(pair_s, labels, "Pairwise γ (sd)")


    out=compute_all_gamma_summaries(data)
    json.dump(out,open("all_gamma_summaries.json","w"))
