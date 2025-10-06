import json
import math
import random
import copy
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# ====== Configuration ======
PLACEHOLDER_LABEL = "!!CHOOSE NEW LABEL!!!"

# NOT comments/issues or taxonomy detritus.
keys_to_remove = {
    "DisfluencyErrorType", "StructuralErrorType", "GrammaticalErrorType",
    "OrthographicErrorType", "PhonologicalErrorType", "VisualTrackingErrorType",
    "MixedErrorTaxonomy", "mispronunciation_word", "produced_word",
    "spoken_word", "spoken_words", "issues", "comments"
}

# Reviewers to exclude entirely
REVIEWERS_TO_EXCLUDE = {70585}

# Optional known user objects (fill these if you have them)
USER_OVERRIDES: Dict[int, Dict[str, Any]] = {
    70293: {"id": 70293, "email": "liu.ying@ufl.edu", "first_name": "", "last_name": ""},
}

# ---------------- Utilities ----------------
def _task_annotator_ids(task: Dict[str, Any]) -> Set[int]:
    """All annotator ids who worked on this task (from annotations[].completed_by.id)."""
    ids = set()
    for ann in task.get("annotations", []):
        cb = ann.get("completed_by") or {}
        if isinstance(cb, dict) and "id" in cb and cb["id"] is not None:
            ids.add(cb["id"])
    return ids

def _task_reviewer_objs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get full reviewer objects (dicts with id/email/first_name/last_name) from reviews[].created_by,
    excluding any whose id is in REVIEWERS_TO_EXCLUDE.
    """
    out = []
    for ann in task.get("annotations", []):
        for rv in ann.get("reviews", []) or []:
            created_by = rv.get("created_by") or {}
            if not isinstance(created_by, dict):
                continue
            rid = created_by.get("id")
            if rid is None or rid in REVIEWERS_TO_EXCLUDE:
                continue
            out.append({
                "id": rid,
                "email": created_by.get("email", ""),
                "first_name": created_by.get("first_name", ""),
                "last_name": created_by.get("last_name", ""),
            })
    return out

def _task_reviewer_ids(task: Dict[str, Any]) -> Set[int]:
    """Convenience: set of reviewer ids (post-exclusion)."""
    return {obj["id"] for obj in _task_reviewer_objs(task)}

def _is_numeric(x) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False

def _id_in_ignore(task: Dict[str, Any], ignore_ids: Set[Any]) -> bool:
    return task.get("id") in ignore_ids or task.get("__id__") in ignore_ids

def _data_has_blank_or_null(obj: Any) -> bool:
    """
    True if any value under task['data'] is a blank string (after strip) or None.
    Booleans (including False) are allowed.
    """
    if obj is None:
        return True
    if isinstance(obj, str):
        return obj.strip() == ""
    if isinstance(obj, (list, tuple)):
        return any(_data_has_blank_or_null(v) for v in obj)
    if isinstance(obj, dict):
        return any(_data_has_blank_or_null(v) for v in obj.values())
    return False

def _find_user_obj_in_task(task: Dict[str, Any], user_id: int) -> Optional[Dict[str, Any]]:
    """
    Search for a full user-like object for user_id inside this task:
      - annotations[].completed_by
      - reviews[].created_by
    Return dict with id/email/first_name/last_name if found, else None.
    """
    for ann in task.get("annotations", []):
        cb = ann.get("completed_by")
        if isinstance(cb, dict) and cb.get("id") == user_id:
            return {
                "id": user_id,
                "email": cb.get("email", ""),
                "first_name": cb.get("first_name", ""),
                "last_name": cb.get("last_name", ""),
            }
    for obj in _task_reviewer_objs(task):
        if obj.get("id") == user_id:
            return obj
    return None

def _completed_by_object(task: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    """
    Build a completed_by dict with required keys. Prefer user object found in task; else USER_OVERRIDES; else minimal default.
    """
    found = _find_user_obj_in_task(task, user_id)
    if found:
        return {
            "id": found.get("id", user_id),
            "email": found.get("email", ""),
            "first_name": found.get("first_name", ""),
            "last_name": found.get("last_name", ""),
        }
    if user_id in USER_OVERRIDES:
        u = USER_OVERRIDES[user_id]
        return {
            "id": u.get("id", user_id),
            "email": u.get("email", "liu.ying@ufl.edu"),
            "first_name": u.get("first_name", ""),
            "last_name": u.get("last_name", ""),
        }
    # final fallback (no task user & no override)
    return {
        "id": user_id,
        "email": "michaelbennie@ufl.edu",
        "first_name": "",
        "last_name": "",
    }

# ---------------- Step 1 ----------------
def compute_tasks_per_annotator(tasks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Count tasks per annotator BEFORE any pruning. If a task has multiple annotators,
    it is counted once for each annotator present.

    Returns a NumPy 2D array: [[annotator_id, total_tasks], ...]
    """
    counts = defaultdict(int)
    for t in tasks:
        for aid in _task_annotator_ids(t):
            counts[aid] += 1
    rows = sorted(counts.items(), key=lambda kv: kv[0])
    return np.array(rows, dtype=object)

# ---------------- Post-filter counts you asked for ----------------
def compute_annotator_counts_after_reviewer(
    tasks: List[Dict[str, Any]],
    *,
    min_task_id: Optional[int] = None,
    ignore_ids: Optional[Iterable[Any]] = None
) -> np.ndarray:
    """
    Count tasks per annotator AFTER applying:
      - ignore_ids / min_task_id,
      - data completeness (no blank/None anywhere in task['data']),
      - exactly one annotator,
      - reviewer filter: <=1 reviewer after exclusions.
    """
    ignore_ids_set = set(ignore_ids or [])
    counts = defaultdict(int)
    for t in tasks:
        if _id_in_ignore(t, ignore_ids_set):
            continue
        tid = t.get("id")
        if min_task_id is not None and tid is not None and _is_numeric(tid) and int(tid) < int(min_task_id):
            continue
        if _data_has_blank_or_null(t.get("data", {})):
            continue
        annotator_ids = _task_annotator_ids(t)
        if len(annotator_ids) != 1:
            continue
        if len(_task_reviewer_ids(t)) > 1:
            continue
        counts[next(iter(annotator_ids))] += 1
    rows = sorted(counts.items(), key=lambda kv: kv[0])
    return np.array(rows, dtype=object)

def compute_reviewer_frequency(
    tasks: List[Dict[str, Any]],
    *,
    min_task_id: Optional[int] = None,
    ignore_ids: Optional[Iterable[Any]] = None
) -> np.ndarray:
    """
    Count frequency of each reviewer id across the same eligible pool as above.
    Only tasks with exactly one reviewer (post-exclusion) are counted; tasks with 0 reviewers are skipped.
    """
    ignore_ids_set = set(ignore_ids or [])
    freq = defaultdict(int)
    for t in tasks:
        if _id_in_ignore(t, ignore_ids_set):
            continue
        tid = t.get("id")
        if min_task_id is not None and tid is not None and _is_numeric(tid) and int(tid) < int(min_task_id):
            continue
        if _data_has_blank_or_null(t.get("data", {})):
            continue
        annotator_ids = _task_annotator_ids(t)
        if len(annotator_ids) != 1:
            continue
        rids = _task_reviewer_ids(t)
        if len(rids) == 1:
            rid = next(iter(rids))
            freq[rid] += 1
    rows = sorted(freq.items(), key=lambda kv: kv[0])
    return np.array(rows, dtype=object)

# ---------------- Step 2 ----------------
def sample_tasks_by_annotator_percent(
    tasks: List[Dict[str, Any]],
    percent: float,
    *,
    annotator_totals_table: np.ndarray,
    min_task_id: Optional[int] = None,
    ignore_ids: Optional[Iterable[Any]] = None,
    seed: Optional[int] = 0
) -> List[Dict[str, Any]]:
    """
    For each annotator present in annotator_totals_table, compute ceil(percent% * total_from_table),
    then sample up to that many tasks attributed to that annotator subject to these EXCLUSIONS:
      - task id < min_task_id (if task.id is numeric),
      - number of annotators != 1,
      - number of reviewers > 1 (after reviewer exclusions),
      - id in ignore_ids,
      - ANY blank-string or null value anywhere in task['data'].
    """
    if seed is not None:
        random.seed(seed)

    ignore_ids_set = set(ignore_ids or [])
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for t in tasks:
        if _id_in_ignore(t, ignore_ids_set):
            continue
        tid = t.get("id")
        if min_task_id is not None and tid is not None and _is_numeric(tid) and int(tid) < int(min_task_id):
            continue
        if _data_has_blank_or_null(t.get("data", {})):
            continue

        annotator_ids = _task_annotator_ids(t)
        if len(annotator_ids) != 1:
            continue

        if len(_task_reviewer_ids(t)) > 1:
            continue

        buckets[next(iter(annotator_ids))].append(t)

    selected: List[Dict[str, Any]] = []
    for row in annotator_totals_table:
        annotator_id, total_from_table = row
        total_from_table = int(total_from_table)

        need = math.ceil((percent / 100.0) * total_from_table)
        pool = buckets.get(annotator_id, [])
        if not pool:
            continue

        k = min(need, len(pool))
        chosen = random.sample(pool, k)
        selected.extend(copy.deepcopy(chosen))

    return selected

# ---------------- Step 3 ----------------
def _make_placeholder_issues_textarea() -> Dict[str, Any]:
    # Keep the exact key spellings you specified (including "Reviwer")
    return {
        "id": "0OYVguZmSW",
        "type": "textarea",
        "value": {"text": ["Placeholder for Reviwer"]},
        "origin": "manual",
        "to_name": "commentHeader",
        "from_name": "issues"
    }

def _scrub_and_placeholder_results_for_prediction(src_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build pre-annotation results by deep-copying original 'result' items, removing
    unwanted keys and replacing label selections with PLACEHOLDER_LABEL.
    """
    out: List[Dict[str, Any]] = []
    for r in src_results:
        r_copy = copy.deepcopy(r)
        if r_copy.get("from_name") in keys_to_remove:
            continue
        if "value" in r_copy and isinstance(r_copy["value"], dict):
            for k in list(r_copy["value"].keys()):
                if k in keys_to_remove:
                    del r_copy["value"][k]
        if r_copy.get("type") == "labels" and "value" in r_copy and "labels" in r_copy["value"]:
            r_copy["value"]["labels"] = [PLACEHOLDER_LABEL]
        out.append(r_copy)
    return out

def create_preannotation_and_fillers(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the original annotations, add one predictions entry with placeholder labels,
    and add ONE filler annotation:
      - If there is a (single) reviewer (post-exclusion), attribute the filler to that reviewer (full user obj).
      - Otherwise attribute it to user 70293 (full user obj via override/minimal).
    """
    t = copy.deepcopy(task)

    base_ann = None
    for ann in t.get("annotations", []):
        if isinstance(ann, dict) and isinstance(ann.get("result"), list):
            base_ann = ann
            break

    src_results: List[Dict[str, Any]] = base_ann.get("result", []) if base_ann else []
    pre_results = _scrub_and_placeholder_results_for_prediction(src_results)

    t.setdefault("predictions", [])
    t["predictions"].append({
        "model_version": "preannotation_v2",
        "result": pre_results
    })

    reviewer_ids = _task_reviewer_ids(t)
    filler_user_id = next(iter(reviewer_ids)) if len(reviewer_ids) >= 1 else 70293
    filler_completed_by = _completed_by_object(t, filler_user_id)

    filler_ann = {
        "completed_by": filler_completed_by,
        "result": [_make_placeholder_issues_textarea()]
    }

    t.setdefault("annotations", []).append(filler_ann)
    t["total_annotations"] = len(t.get("annotations", []))
    t["total_predictions"] = len(t.get("predictions", []))
    return t

def _update_task_counts_in_place(tasks: List[Dict[str, Any]]) -> None:
    for tt in tasks:
        tt["total_annotations"] = len(tt.get("annotations", []))
        tt["total_predictions"] = len(tt.get("predictions", []))

# ---------------- Orchestration ----------------
def generate_cross_annotation_json_v2(
    input_file_path: Path,
    output_file_path: Path,
    percent: float,
    *,
    min_task_id: Optional[int] = None,
    ignore_ids: Optional[Iterable[Any]] = None,
    seed: Optional[int] = 0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads JSON, computes per-annotator totals (Step 1),
    samples tasks per annotator at ceil(percent%) using totals from Step 1 (Step 2),
    attaches predictions + filler annotations (Step 3),
    and writes the processed sample to output_file_path.

    Returns:
      (annotator_totals_table_before, processed_sample_tasks)
    """
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of task objects.")

    totals_table = compute_tasks_per_annotator(data)

    sampled_tasks = sample_tasks_by_annotator_percent(
        data,
        percent,
        annotator_totals_table=totals_table,
        min_task_id=min_task_id,
        ignore_ids=ignore_ids,
        seed=seed
    )

    processed = [create_preannotation_and_fillers(t) for t in sampled_tasks]
    _update_task_counts_in_place(processed)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    return totals_table, processed

def compute_annotator_counts_in_sample(sampled_tasks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Count how many tasks from each annotator are in the FINAL SAMPLE.
    Assumes each task has exactly one annotator (because the sampler enforces that).
    """
    counts = defaultdict(int)
    for t in sampled_tasks:
        annotator_ids = _task_annotator_ids(t)

        counts[next(iter(annotator_ids))] += 1
    rows = sorted(counts.items(), key=lambda kv: kv[0])
    return np.array(rows, dtype=object)



# -------- Example CLI usage --------
if __name__ == "__main__":
    INPUT_JSON = Path("../annotationData/words/export_157618_project-157618-at-2025-10-06-08-37-e2ae7485.json")
    OUTPUT = Path("../processed_data/cross-annotations-v2.json")

    table_before, sample = generate_cross_annotation_json_v2(
        input_file_path=INPUT_JSON,
        output_file_path=OUTPUT,
        percent=2,
        min_task_id=None,
        ignore_ids=None,
        seed=42
    )

    full_data = json.load(open(INPUT_JSON, "r", encoding="utf-8"))
    annotator_counts_after = compute_annotator_counts_after_reviewer(full_data)
    reviewer_freq = compute_reviewer_frequency(full_data)
    sample_counts = compute_annotator_counts_in_sample(sample)

    print("Annotator task counts (before pruning):")
    for annotator_id, total in table_before:
        print(f"  annotator {annotator_id}: {total} tasks")

    print("\nAnnotator task counts (after reviewer/data filters):")
    for annotator_id, total in annotator_counts_after:
        print(f"  annotator {annotator_id}: {total} tasks")

    print("\nAnnotator task counts (in FINAL SAMPLE):")
    for annotator_id, total in sample_counts:
        print(f"  annotator {annotator_id}: {total} sampled tasks")

    print("\nReviewer frequency (post-exclusion):")
    for reviewer_id, total in reviewer_freq:
        print(f"  reviewer {reviewer_id}: {total} tasks")

    print(f"\nWrote {len(sample)} sampled+processed tasks to {OUTPUT}")
