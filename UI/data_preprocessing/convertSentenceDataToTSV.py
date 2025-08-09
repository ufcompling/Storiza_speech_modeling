import json
import pandas as pd


def convert_sentences_to_tsv(input_file: str, output_file: str):
    """Convert Label‑Studio sentence annotations to a TSV covering ALL tasks."""
    url_prefix = "https://2025storiza.michaelbennie.org/"

    with open(input_file, "r", encoding="utf‑8") as f:
        data = json.load(f)

    rows = []

    for task in data:
        task_data = task.get("data", {})
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

        current = None
        for ann in results:
            a_type = ann["type"]
            a_val = ann["value"]

            # ── new sentence ──────────────────────────────────────────────
            if a_type == "labels" and ann["from_name"] == "SentenceLabel":
                # flush previous
                if current:
                    if current["start_time"] is not None and current["end_time"] is not None:
                        current["segment_time"] = current["end_time"] - current["start_time"]
                    rows.append(current)

                start, end = a_val.get("start"), a_val.get("end")
                current = {
                    # sentence‑level
                    "audio": audio,
                    "start_time": start,
                    "end_time": end,
                    "goldStandard": "",
                    "actual": "",
                    "repeated": False,
                    "runon": False,
                    "nonchild": False,
                    # task‑level
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

            # ── gold‑standard text ────────────────────────────────────────
            elif a_type == "choices" and ann["from_name"] == "SentenceSelect":
                current["goldStandard"] = a_val["choices"][0]

            # ── child’s production ────────────────────────────────────────
            elif a_type == "textarea" and ann["from_name"] == "Sentence":
                current["actual"] = a_val["text"][0]

            # ── sentence flags ────────────────────────────────────────────
            elif a_type == "choices" and ann["from_name"] == "sentenceIssues":
                issues = set(a_val["choices"])
                current["repeated"] = "repeated" in issues
                current["runon"] = "runon" in issues
                current["nonchild"] = "Not" in issues

        # flush last sentence of the task
        if current:
            if current["start_time"] is not None and current["end_time"] is not None:
                current["segment_time"] = current["end_time"] - current["start_time"]
            rows.append(current)

    # save TSV
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    convert_sentences_to_tsv(
        "../annotationData/sentences/export_157513_project-157513-at-2025-06-12-02-07-aba526d5.json",
        "../processed_data/sentenceLabels.tsv",
    )
