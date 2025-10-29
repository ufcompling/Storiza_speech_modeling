import re
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Counter
from copy import deepcopy
from Labels import *

def _parse_dt(s: Optional[str]) -> datetime:
    if not s:
        return datetime(1970, 1, 1)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1)

def _load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unrecognized JSON format for annotations export.")

def _group_tasks_by_audio(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        audio = t.get("data", {}).get("audio")
        if not audio:
            continue
        annotations=t["annotations"]
        out.setdefault(audio, []).extend(annotations)
    return out

def _flatten_annotations_per_audio(tasks_for_audio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for t in tasks_for_audio:
        anns = t["result"]
        for a in anns:
            ann = deepcopy(a)
            ann["_task_id"] = t.get("id")
            ann["_audio"] = t.get("data", {}).get("audio")
            flat.append(ann)
    return flat

def _filter_by_count(anns: List[Dict[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    return anns if len(anns) == expected_count else []

def _keep_x_newest(anns: List[Dict[str, Any]], x: int) -> List[Dict[str, Any]]:
    if x <= 0:
        return anns
    sorted_anns = sorted(anns, key=lambda a: _parse_dt(a.get("created_at")), reverse=True)
    return sorted_anns[:x]

def _group_results_by_region_id(result_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in result_list:
        rid = r.get("id")
        if rid is None:
            continue
        groups.setdefault(rid, []).append(r)
    return groups

def _init_bool_map(names: List[str]) -> Dict[str, bool]:
    return {n: False for n in names}

def _merge_text(values: List[str]) -> Optional[str]:
    seen = set()
    merged = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            merged.append(v)
            seen.add(v)
    return " | ".join(merged) if merged else None

def _build_general_and_specific_for_region(region_items: List[Dict[str, Any]]) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    general_map = _init_bool_map(TOP_LEVEL_LABELS)
    specific_map = _init_bool_map(SPECIFIC_LABELS)

    top_level_selected: List[str] = []
    for it in region_items:
        if it.get("type") == "labels" and it.get("from_name") == "WordAnnotation":
            label_vals=it.get("value", {}).get("labels", [])
            for i in range(len(label_vals)):
                if label_vals[i] in NAME_NORMALIZING_MAP:
                    label_vals[i]=NAME_NORMALIZING_MAP[label_vals[i]]
            top_level_selected.extend(label_vals)

    if top_level_selected and "Mixed Error" not in top_level_selected:
        for lab in top_level_selected:
            if lab in general_map:
                general_map[lab] = True

    for it in region_items:
        if it.get("type") == "choices":
            from_name = it.get("from_name")
            selected = it.get("value", {}).get("choices", [])
            for i in range(len(selected)):
                if selected[i] in NAME_NORMALIZING_MAP:
                    selected[i]=NAME_NORMALIZING_MAP[selected[i]]
            for s in selected:
                if s in specific_map:
                    specific_map[s] = True
            family_to_top = {
                "PhonologicalErrorType": "Phonological Error",
                "OrthographicErrorType": "Orthographic Error",
                "GrammaticalErrorType": "Grammatical Error",
                "StructuralErrorType": "Structural Error",
                "VisualTrackingErrorType": "Visual Tracking Error",
                "DisfluencyErrorType": "Disfluency Error",
            }
            fam = family_to_top.get(from_name)
            if fam and "Mixed Error" not in top_level_selected and fam in general_map:
                general_map[fam] = True

    for it in region_items:
        if it.get("type") == "taxonomy" and it.get("from_name") == "MixedErrorTaxonomy":
            tax = it.get("value", {}).get("taxonomy", [])
            for node in tax:
                if not node:
                    continue
                if len(node) == 1:
                    fam = node[0]
                    top = TAXONOMY_TO_TOPLEVEL.get(fam)
                    if top and top in general_map:
                        general_map[top] = True
                else:
                    fam, leaf = node[0], node[1]
                    top = TAXONOMY_TO_TOPLEVEL.get(fam, TAXONOMY_TO_TOPLEVEL.get(leaf))
                    if top and top in general_map:
                        general_map[top] = True
                    if leaf in specific_map:
                        specific_map[leaf] = True
                    if fam in specific_map:
                        specific_map[fam] = True

    if "Mixed Error" in general_map:
        general_map["Mixed Error"] = False

    return general_map, specific_map


def clean_ipa(ipa_list: List[str]) -> List[str]:
    """
    Normalizes the first IPA string in `ipa_list`:
      - strips slashes
      - removes tie-bars in affricates (͡, ͜) and collapses to digraphs
      - expands rhotized vowels (precomposed ɚ/ɝ and vowel + ˞) to vowel + ɹ
      - converts "upwards r" variants to ɹ

    Returns a single-element list with the cleaned string (mirrors original API).
    """
    if not ipa_list:
        return ipa_list

    s = ipa_list[0]

    # 1) strip surrounding slashes if present
    if "/" in s:
        s = s.replace("/", "")
    if "[" in s:
        s = s.replace("[", "")
    if "]" in s:
        s = s.replace("]", "")
    if "\"" in s:
        s = s.replace("\"", "")

    # 2) remove tie bars so affricates become plain digraphs (e.g., t͡s -> ts, d͜z -> dz)
    #    U+0361 COMBINING DOUBLE INVERTED BREVE (͡), U+035C COMBINING DOUBLE BREVE BELOW (͜)
    s = s.replace("\u0361", "").replace("\u035C", "")

    # Keep explicit affricate simplifications too (no-op now, but harmless and explicit)
    s = s.replace("d͡ʒ", "dʒ").replace("t͡ʃ", "tʃ")

    # 3) RHOTICS: expand rhotized vowels to vowel + ɹ
    #    - precomposed: ɚ -> əɹ, ɝ -> ɜɹ (approximation that preserves rhoticity)
    s = s.replace("ɚ", "əɹ").replace("ɝ", "ɜɹ")

    #    - combining rhotic hook: any vowel + ˞ (U+02DE) -> vowel + ɹ
    #      We'll match a fairly broad set of IPA vowel symbols.
    IPA_VOWELS = "aeiou" \
                 "ɑæɐəɜɛeɪioɔuʊʌɒœøɯyɨʉɘɵɤʏɶɞ"

    # replace VOWEL + ˞ with VOWEL + ɹ
    s = re.sub(fr"([{IPA_VOWELS}])\u02DE", r"\1ɹ", s)

    #    - superscript r after a vowel: V + ʳ -> V + ɹ
    s = re.sub(fr"([{IPA_VOWELS}])\u02B3", r"\1ɹ", s)  # U+02B3 = ʳ

    # 4) Normalize all "upwards r" variants to ɹ
    #    (choose what you want included here; this set is intentionally broad)
    R_VARIANTS = [
        "r",  # alveolar trill
        "ɻ",  # retroflex approximant
        "ʳ",  # modifier letter small r (superscript)
        "ʴ",  # modifier letter small turned r with hook
    ]
    for rv in R_VARIANTS:
        s = s.replace(rv, "ɹ")

    # 5) (optional) seperate any accidental double ɹɹ that could arise from replacements (rare)
    s = re.sub("ɹ{2,}", "ɹɹ.", s)

    # Consolidating prolongation
    if 'ː' in s:
        s = s.replace('ː', ':')

    if 'β' in s:
        s = s.replace('β', 'b')
    if 'ʋ' in s:
        s = s.replace('ʋ', 'ɹ')
    if 'ʑ' in s:
        s = s.replace('ʑ', 'ʒ')
    if 'ɫ' in s:
        s = s.replace('ɫ', 'l')
    if 'ɝ' in s:
        s = s.replace('ɝ', 'ɜɹ')

    # Diacritics
    if "ʰ" in s:
        s = s.replace("ʰ", '')
    if "ˈ" in s:
        s = s.replace("ˈ", '')
    if "ˌ" in s:
        s = s.replace("ˌ", '')
    if "ː" in s:
        s = s.replace("ː", '')
    if "̃" in s:
        s = s.replace("̃", '')
    if "̈" in s:
        s = s.replace("̈", '')
    if "̚" in s:
        s = s.replace("̚", '')
    if "̩" in s:
        s = s.replace("̩", '')
    if "̰" in s:
        s = s.replace("̰", '')
    if "̚" in s:
        s = s.replace("̚", '')

    # Changing some vowels
    if 'ɐ' in s:
        index_list = []
        for i in range(len(s)):
            phoneme = s[i]
            if phoneme == 'ɐ':
                index_list.append(i)

        # ɐ followed by ɪ --> a
        # ɐ followed by ʊ --> a
        # ɐ (elsewhere) --> ɑ
        s = list(s)
        follow_check = 0
        for idx in index_list:
            try:
                next_phoneme = index_list[idx + 1]
                if next_phoneme == 'ɪ' or next_phoneme == 'ʊ':
                    print(s)
                    s[idx] = 'a'
                    follow_check += 1
            except:
                pass
        if follow_check == 0:
            for idx in index_list:
                s[idx] = 'ɑ'
        s = ' '.join(s)

    if 'ɒ' in s:
        s = s.replace('ɒ', 'ɑ')

    return [s]


def decide_actual_production(error_category: str, intended: str, produced: str, ipa: List[str]) -> str:
	if error_category == 'Correct': #bruh
		return intended, 'word'
	if intended and not produced and not ipa:
		return intended, 'word'
	if intended and not produced and ipa:
		return ipa[0], 'ipa'
	if intended and produced and not ipa:
		return produced, 'word'
	if intended and produced and ipa:
		return produced, 'word'
	if (not intended) and (not produced) and ipa:
		return ipa[0], 'ipa'
	if (not intended) and produced and not ipa:
		return produced, 'word'
	if (not intended) and produced and ipa:
		return produced, 'word'
	return '', ''

def _transform_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    res = annotation.get("result", [])
    regional = _group_results_by_region_id(res)

    simplified_items = []
    for rid, items in regional.items():
        starts, ends = [], []
        intended_words, produced_words, ipa_vals, intended_words_multi = [], [], [], []

        for it in items:
            val = it.get("value", {})
            if "start" in val:
                starts.append(val.get("start"))
            if "end" in val:
                ends.append(val.get("end"))
            if it.get("type") == "textarea":
                txts = val.get("text", [])
                if it.get("from_name") == "spoken_word":
                    intended_words.extend(txts)
                elif it.get("from_name") == "produced_word":
                    produced_words.extend(txts)
                elif it.get("from_name") == "mispronunciation_word":
                    ipa_vals.extend(txts)
                elif it.get("from_name") == "spoken_words":
                    intended_words_multi.extend(txts)

        start = min(starts) if starts else None
        end = max(ends) if ends else None

        general_map, specific_map = _build_general_and_specific_for_region(items)

        item_out = {
            "start": start,
            "end": end,
            "general_label_type": {
                "labels": TOP_LEVEL_LABELS,
                "selected": [bool(general_map[l]) for l in TOP_LEVEL_LABELS],
            },
            "specific_label_type": {
                "labels": SPECIFIC_LABELS,
                "selected": [bool(specific_map[l]) for l in SPECIFIC_LABELS],
            },
        }


        iw = _merge_text(intended_words)
        pw = _merge_text(produced_words)
        ipa = _merge_text(ipa_vals)
        iws = _merge_text(intended_words_multi)
        if iw is not None:
            item_out["intended_word"] = iw
        if pw is not None:
            item_out["produced_word"] = pw
        if ipa is not None:
            item_out["mispronunciation_ipa"] = clean_ipa([ipa])[0]
        if iws is not None:
            item_out["intended_word"] = iws

        simplified_items.append(item_out)

    simplified_items.sort(key=lambda x: (float("inf") if x["start"] is None else x["start"],
                                         float("inf") if x["end"] is None else x["end"]))

    new_ann = deepcopy(annotation)
    new_ann["result"] = simplified_items
    return new_ann

def _build_output_for_audio(annotations_flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ann in annotations_flat:
        out.append(_transform_annotation(ann))
    return out

def group_annotations(
    json_path: str,
    leave_x_newest_annotations: Optional[int] = None,
    keep_only_audio_with_x_annotations: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    path = Path(json_path)
    tasks = _load_json(path)
    by_audio = _group_tasks_by_audio(tasks)

    result_mapping: Dict[str, List[Dict[str, Any]]] = {}

    for audio, task_list in by_audio.items():

        task_list = _dedupe_by_completed_by_keep_newest(task_list)

        if keep_only_audio_with_x_annotations is not None: # this is used to only keep the items that were fully annotated//a
            task_list = _filter_by_count(task_list, keep_only_audio_with_x_annotations)

        if leave_x_newest_annotations and leave_x_newest_annotations > 0: #this is used to only keep the cross annotation
            task_list = _keep_x_newest(task_list, leave_x_newest_annotations)


        if not task_list:
            continue

        transformed = _build_output_for_audio(task_list)
        result_mapping[audio] = transformed

    return result_mapping

# -------------------- Printing HELPERS --------------------

def _print_heading(title: str) -> None:
    print(title)
    print("-" * max(3, len(title)))

def _summary_line(name: str, mapping: Dict[str, List[Dict[str, Any]]]) -> None:
    num_audios = len(mapping)
    num_annotations = sum(len(v) for v in mapping.values())
    print(f"{name:<45} Audios: {num_audios:<6} Total Annotations: {num_annotations}")



# -------------------- generate cross annotation dicst --------------------

def join_and_concat(
    a: Dict[str, List[Dict[str, Any]]],
    b: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    INNER JOIN on keys: only keys present in BOTH a and b.
    Value for each shared key = concatenation of the lists a[k] + b[k].
    """
    shared = set(a.keys()) & set(b.keys())
    out={k: list(a[k]) + list(b[k]) for k in shared}
    for key in out.keys():
        out[key]=_dedupe_by_completed_by_keep_newest(out[key])

    return out

def union_concat(
    a: Dict[str, List[Dict[str, Any]]],
    b: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    UNION on keys: keys in either dict. If key in both, concatenate lists.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k in set(a.keys()) | set(b.keys()):
        out[k] = list(a.get(k, [])) + list(b.get(k, []))

    for key in out.keys():
        out[key]=_dedupe_by_completed_by_keep_newest(out[key])


    return out

def _dedupe_by_completed_by_keep_newest(anns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse multiple annotations from the same 'completed_by' dict to the newest one.
    Equality is based on the full completed_by dict (same values inside).
    Recency uses 'created_at' (falls back to id when needed).
    """
    buckets: Dict[str, Dict[str, Any]] = {}

    def _recency_key(a: Dict[str, Any]) -> Tuple[datetime, int]:
        dt = _parse_dt(a.get("created_at"))
        # id is a stable secondary tiebreaker if timestamps tie or are missing
        aid = int(a.get("id", 0)) if isinstance(a.get("id"), (int, str)) and str(a.get("id")).isdigit() else 0
        return (dt, aid)

    for a in anns:
        key_obj = a.get("completed_by", {})
        # stable string key from the entire dict content
        key_str = json.dumps(key_obj, sort_keys=True)
        best = buckets.get(key_str)
        if best is None or _recency_key(a) > _recency_key(best):
            buckets[key_str] = a

    return list(buckets.values())


def generate_cross_annotation_dicts(
    annotations_v2_json_path: str = "../../processed_annotations/export_194012_project-194012-at-2025-10-26-23-02-5adfae3c.json",
    annotations_v1_json_path: str = "../../processed_annotations/export_178326_project-178326-at-2025-09-29-04-04-8271fa09.json",
    annotations_original_json_path: str = "../../processed_annotations/export_157618_project-157618-at-2025-10-26-23-01-53494ae5.json",
):
    """
    Runs the grouping with  specified filters, prints a summary table,
    and returns a dict of all four mappings.
    """
    cross_annotations_v2 = group_annotations(
        annotations_v2_json_path,
        leave_x_newest_annotations=1,
        keep_only_audio_with_x_annotations=3,
    )
    for key in cross_annotations_v2.keys():
        cross_annotations_v2[key]=_dedupe_by_completed_by_keep_newest(cross_annotations_v2[key])


    cross_annotations_v1 = group_annotations(
        annotations_v1_json_path,
        leave_x_newest_annotations=1,
        keep_only_audio_with_x_annotations=2,
    )
    for key in cross_annotations_v1.keys():
        cross_annotations_v1[key]=_dedupe_by_completed_by_keep_newest(cross_annotations_v1[key])

    original_annotations = group_annotations(
        annotations_original_json_path,
    )
    for key in original_annotations.keys():
        original_annotations[key]=_dedupe_by_completed_by_keep_newest(original_annotations[key])


    original_annotations_accidentally_crossed = group_annotations(
        annotations_original_json_path,
        leave_x_newest_annotations=2,
        keep_only_audio_with_x_annotations=2,  # ones that *were* accidentally cross-annotated
    )
    for key in original_annotations_accidentally_crossed.keys():
        original_annotations_accidentally_crossed[key]=_dedupe_by_completed_by_keep_newest(original_annotations_accidentally_crossed[key])


    _print_heading("Number of audios annotated and total annotations for each (raw groups)")
    _summary_line("cross_annotations_v1", cross_annotations_v1)
    _summary_line("cross_annotations_v2", cross_annotations_v2)
    _summary_line("original_annotations", original_annotations)
    _summary_line("original_annotations_accidentally_crossed", original_annotations_accidentally_crossed)
    print("-" * 59)

    return {
        "cross_annotations_v1": cross_annotations_v1,
        "cross_annotations_v2": cross_annotations_v2,
        "original_annotations": original_annotations,
        "original_annotations_accidentally_crossed": original_annotations_accidentally_crossed,
    }

def _count_labels(mapping: Dict[str, List[Dict[str, Any]]]) -> Tuple[Counter, Counter]:
    """
    Count how many times each general and specific label occurs
    across all annotations in this mapping.
    """
    general_counter = Counter()
    specific_counter = Counter()

    for audio, annotations in mapping.items():
        for ann in annotations:
            for region in ann.get("result", []):
                gen_labels = region.get("general_label_type", {})
                spec_labels = region.get("specific_label_type", {})

                gen_names = gen_labels.get("labels", [])
                gen_selected = gen_labels.get("selected", [])
                spec_names = spec_labels.get("labels", [])
                spec_selected = spec_labels.get("selected", [])

                for name, selected in zip(gen_names, gen_selected):
                    if selected:
                        general_counter[name] += 1

                for name, selected in zip(spec_names, spec_selected):
                    if selected:
                        specific_counter[name] += 1

    return general_counter, specific_counter


def _print_label_counts(name: str, mapping: Dict[str, List[Dict[str, Any]]]) -> None:
    general_counts, specific_counts = _count_labels(mapping)
    print(f"\nLabel distribution for {name}:")
    print("-" * (25 + len(name)))
    print("Top-level (general) labels:")
    for label, count in general_counts.most_common():
        print(f"  {label:<30} {count}")
    print("\nSpecific labels:")
    for label, count in specific_counts.most_common():
        print(f"  {label:<30} {count}")
    print("-" * (25 + len(name)))


def generate_combined_cross_annotation_dicts(
        annotations_v2_json_path: str = "../../processed_annotations/export_194012_project-194012-at-2025-10-26-23-02-5adfae3c.json",
        annotations_v1_json_path: str = "../../processed_annotations/export_178326_project-178326-at-2025-09-29-04-04-8271fa09.json",
        annotations_original_json_path: str = "../../processed_annotations/export_157618_project-157618-at-2025-10-26-23-01-53494ae5.json",

):

    # Generate the four base dicts and print their summaries
    dicts = generate_cross_annotation_dicts()

    cross_annotations_v1 = dicts["cross_annotations_v1"]
    cross_annotations_v2 = dicts["cross_annotations_v2"]
    original_annotations = dicts["original_annotations"]
    original_annotations_accidentally_crossed = dicts["original_annotations_accidentally_crossed"]

    # Build the requested combined views:
    # all_cross = (cross_v1 + cross_v2) + original_annotations  [UNION-combine]
    # Build the requested combined views using INNER JOINS with original
    # 1) all_cross = inner join( union(v1, v2), original )
    merged_v1_v2 = union_concat(cross_annotations_v1, cross_annotations_v2)


    all_cross = join_and_concat(merged_v1_v2, original_annotations)


    # 2) v1_cross = inner join( v1, original )
    v1_cross = join_and_concat(cross_annotations_v1, original_annotations)

    # 3) v2_cross = inner join( v2, original )
    v2_cross = join_and_concat(cross_annotations_v2, original_annotations)

    v1_v2_overlap = join_and_concat(join_and_concat(cross_annotations_v1, cross_annotations_v2),original_annotations)

    # 4) combination of all cross-annotations
    all_cross_overlap = union_concat(all_cross, original_annotations_accidentally_crossed)

    _print_heading("Combined stats")
    _summary_line("all_cross ( (v1 ∪ v2) ⋂ original )", all_cross)
    _summary_line("v1_cross ( v1 ⋂ original )", v1_cross)
    _summary_line("v2_cross ( v2 ⋂ original )", v2_cross)
    _summary_line("overlap in cross annotations ( v2 ⋂ v1 ⋂ original)", v1_v2_overlap)
    _summary_line("original with >1 annotations", original_annotations_accidentally_crossed)
    _summary_line("all_cross_overlap (((v1 ∪ v2) ⋂ original) ∪ original with >1 annotations )", all_cross_overlap)
    print("-" * 59)

    # --- Label frequency summaries ---
    for name, mapping in {
        "all_cross_overlap": all_cross_overlap,
    }.items():
        _print_label_counts(name, mapping)


    return {
        "all_cross": all_cross,
        "v1_cross": v1_cross,
        "v2_cross": v2_cross,
        "accidentally_crossed": original_annotations_accidentally_crossed,
        "v1_v2_overlap": v1_v2_overlap,
        "all_cross_overlap": all_cross_overlap,
    }

if __name__ == "__main__":
    generate_combined_cross_annotation_dicts(
        annotations_v2_json_path = "../../processed_annotations/export_194012_project-194012-at-2025-10-26-23-02-5adfae3c.json",
        annotations_v1_json_path = "../../processed_annotations/export_178326_project-178326-at-2025-09-29-04-04-8271fa09.json",
        annotations_original_json_path = "../../processed_annotations/export_157618_project-157618-at-2025-10-26-23-01-53494ae5.json",
    )



