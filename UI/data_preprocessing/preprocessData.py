import json
from pathlib import Path

import pandas as pd
from typing import *


def strip_quotes(s: str) -> str:
    """Normalise straight/curly quotes and trim."""
    return s.replace("“", "\"").replace("”", "\"").replace("¡¾4:4?source¡¿","").replace("Pok¨¦mon","Pokemon").replace("‘", "'").replace("’", "'").strip()


def combine_error_pairs(
    sentences: List[Dict[str, Any]],
    error_entries: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Update the goldStandard of sentence pairs based on provided error entries.
    Ensures all error entries are used exactly once and warns if second_part
    is already present in any sentence.
    """
    if not sentences or not error_entries:
        return sentences

    if isinstance(error_entries, dict):
        error_entries = [error_entries]

    used_error_indices = set()

    i = 0
    used_i=set()
    while i < len(sentences):
        cur = sentences[i]
        for idx, err in enumerate(error_entries):
            first = strip_quotes(err["first_sentence_to_combine"])
            second_part = strip_quotes(err["second_sentence_to_combine"])

            # Warn if second_part appears anywhere else in the task
            for j, s in enumerate(sentences):
                gold = strip_quotes(s.get("goldStandard", ""))
                if j != i and j not in used_i and second_part and second_part in gold:
                    print(f"⚠️  Potential duplication of second_part in sentence {j}:\n"
                          f"    → second_part: '{second_part}'\n"
                          f"    → matched goldStandard: '{gold}'")

            # Check for match and update
            if strip_quotes(cur.get("goldStandard", "")) == first:
                cur["goldStandard"] = f"{first} {second_part}".strip()
                used_error_indices.add(idx)
                used_i.add(i)
                break
        i += 1

    # After processing, ensure all error entries were used
    unused = [error_entries[i] for i in range(len(error_entries)) if i not in used_error_indices]
    if unused:
        print(f"⚠️ {len(unused)} error entries were not used:")
        for entry in unused:
            print(json.dumps(entry, indent=2, ensure_ascii=False))

    return sentences


def correctly_split_errors(
    sentences: List[Dict[str, Any]],
    split_entries: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Fix “type-5” segmentation errors:
    For each combined sentence listed in *split_entries*,
    find all matching rows in order,
    then replace them with the corresponding split parts (supports 2+).
    """
    if not sentences or not split_entries:
        return sentences

    if isinstance(split_entries, dict):
        split_entries = [split_entries]

    for item in split_entries:
        for combined, parts in item.items():
            combined_clean = strip_quotes(combined)
            split_parts = [strip_quotes(parts[k]) for k in sorted(parts.keys(), key=lambda x: int(x))]

            # Find all rows that match this combined sentence exactly
            matching_idxs = [
                i
                for i, row in enumerate(sentences)
                if strip_quotes(row.get("goldStandard", "")) == combined_clean
            ]

            expected_count = len(split_parts)

            if len(matching_idxs) < expected_count:
                print(
                    f"⚠️  Expected {expected_count} rows for combined sentence but found {len(matching_idxs)}:\n"
                    f"    '{combined_clean}'"
                )
                continue

            for idx, part in zip(matching_idxs, split_parts):
                sentences[idx]["goldStandard"] = part

    return sentences



def convert_sentences_to_tsv(input_file: str, output_file: str,error_map_file: str,split_map_file:str ):
    """Convert Label‑Studio sentence annotations to a TSV covering ALL tasks."""
    url_prefix = "https://2025storiza.michaelbennie.org/"

    with open(input_file, "r", encoding="utf‑8") as f:
        data = json.load(f)

    error_map: Dict[str, Any] = json.loads(Path(error_map_file).read_text(encoding="utf-8"))
    split_map: Dict[str, Any] = json.loads(Path(split_map_file).read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []




    for task in data:
        task_data = task.get("data", {})
        sentence_level_id=str(task.get("id", ""))
        audio_full = task_data.get("audio", "")
        audio = (
            audio_full[len(url_prefix) :]
            if audio_full.startswith(url_prefix)
            else audio_full
        )

        # Skip tasks without annotations
        if not task.get("annotations"):
            continue

        annot = task["annotations"][0]  # keep the first completed annotation
        annotator_id = annot.get("completed_by", {}).get("id", "")
        results = annot["result"]
        if len(task["annotations"])>1:
                print("multiple annotations found!!!",sentence_level_id)




        current = None
        task_sentences = []


        for ann in results:
            a_type = ann["type"]
            a_val = ann["value"]

            # ── new sentence ──────────────────────────────────────────
            if a_type == "labels" and ann["from_name"] == "SentenceLabel":
                # flush previous
                if current and current["goldStandard"] not in ["", "Other", "Other"]:
                    if current["start_time"] is not None and current["end_time"] is not None:
                        current["segment_time"] = current["end_time"] - current["start_time"]

                    current["goldStandard"] = strip_quotes(current["goldStandard"].strip())

                    task_sentences.append(current)

                start, end = a_val.get("start"), a_val.get("end")
                current = {
                    # sentence-level
                    "sentence_level_id": sentence_level_id,
                    "audio": audio,
                    "start_time": start,
                    "end_time": end,
                    "goldStandard": "",
                    "actual": "",
                    "repeated": False,
                    "runon": False,
                    "nonchild": False,
                    # task-level
                    "annotator_id": annotator_id,
                    "grade": task_data.get("grade", ""),
                    "sound": task_data.get("sound", ""),
                    "title": task_data.get("title", ""),
                    "topic": task_data.get("topic", ""),
                    "words": task_data.get("words", ""),
                    "__id__": task_data.get("__id__", ""),
                    "content": task_data.get("content", ""),
                    "time": task_data.get("time", ""),
                    "picture": task_data.get("picture", ""),
                    "userId": task_data.get(
                        "userId (matches the uid in the recording file name)", ""
                    ),
                    # calculated later
                    "segment_time": "",
                }

            # ── gold-standard text ───────────────────────────────────
            elif a_type == "choices" and ann["from_name"] == "SentenceSelect":
                current["goldStandard"] = a_val["choices"][0]

            # ── child’s production ───────────────────────────────────
            elif a_type == "textarea" and ann["from_name"] == "Sentence":
                current["actual"] = a_val["text"][0]

            # ── sentence flags ───────────────────────────────────────
            elif a_type == "choices" and ann["from_name"] == "sentenceIssues":
                issues = set(a_val["choices"])
                current["repeated"] = "repeated" in issues
                current["runon"] = "runon" in issues
                current["nonchild"] = "Not" in issues

        # flush last sentence of the task
        if current and current["goldStandard"] not in ["", "Other", "Other"]:
            if current["start_time"] is not None and current["end_time"] is not None:
                current["segment_time"] = current["end_time"] - current["start_time"]
            current["goldStandard"] = strip_quotes(current["goldStandard"].strip())
            task_sentences.append(current)

        # ── NEW: sort and extend rows ───────────────────────────────────
        task_sentences.sort(
            key=lambda s: (
                s["start_time"] is None,        # None goes last
                s["start_time"] if s["start_time"] is not None else 0
            )
        )




        if str(sentence_level_id)=='188964307' or True:


            # 1️⃣ merge 6b “join pairs”
            task_sentences = combine_error_pairs(task_sentences, error_map.get(sentence_level_id))

            # 2️⃣ split 5-error “over-combined” sentences
            task_sentences = correctly_split_errors(task_sentences, split_map.get(sentence_level_id))

            rows.extend(task_sentences)


    # save TSV
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    convert_sentences_to_tsv(
        "../annotationData/sentences/export_157513_project-157513-at-2025-07-08-04-28-a6ff5f03.json",
        "../processed_data/sentenceLabels.tsv",
        "../ErrorData/6b_errors.json",
                "../ErrorData/5_errors.json",
    )
