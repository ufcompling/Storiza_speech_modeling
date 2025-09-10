import json
from pathlib import Path

def strip_to_prelabels_without_comments(tasks, comment_from_names=("issues",), comment_to_names=("commentHeader",)):
    """
    Given a list of Label Studio tasks, return a new list that keeps only
    `data` and `predictions`, with any comment results removed from predictions.
    """
    cleaned = []
    for task in tasks:
        new_task = {"data": task.get("data", {})}

        preds = task.get("predictions", [])
        new_preds = []
        for pred in preds:
            pred_copy = {k: v for k, v in pred.items() if k != "result"}
            results = pred.get("result", [])
            filtered_results = []
            for r in results:
                if r.get("from_name") in comment_from_names:
                    continue
                if r.get("to_name") in comment_to_names:
                    continue
                filtered_results.append(r)
            pred_copy["result"] = filtered_results
            new_preds.append(pred_copy)

        if preds:
            new_task["predictions"] = new_preds

        cleaned.append(new_task)
    return cleaned


def main(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Handle possible "tasks" wrapper in export
    if isinstance(payload, dict) and "tasks" in payload:
        tasks = payload["tasks"]
    else:
        tasks = payload

    # Process
    cleaned_tasks = strip_to_prelabels_without_comments(tasks)

    # Save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_tasks, f, ensure_ascii=False, indent=2)

    print(f"Cleaned pre-labels saved to {output_path}")


if __name__ == "__main__":
    input_path = "../annotationData/word_cross_annotation/export_178326_project-178326-at-2025-08-11-17-36-add6b2e3.json"
    output_path = "../processed_data/removed_comments_test_crosss_annotations.json"
    main(input_path,output_path)
