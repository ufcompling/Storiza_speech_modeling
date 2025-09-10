import json
from collections import defaultdict
import copy

# Keys to remove from annotation results (by from_name or within value dicts)
keys_to_remove = {
    "DisfluencyErrorType", "StructuralErrorType", "GrammaticalErrorType",
    "OrthographicErrorType", "PhonologicalErrorType", "VisualTrackingErrorType",
    "MixedErrorTaxonomy", "mispronunciation_word", "produced_word",
    "spoken_word", "spoken_words",
    "issues", "comments"  # also drop comment/issue fields
}


def create_cleaned_predictions(task):
    """
    Create cleaned predictions from annotations without modifying original annotations.
    - Skips any result whose from_name is in keys_to_remove (including issues/comments).
    - Removes any fields in keys_to_remove that appear inside result['value'].
    - Replaces 'labels' selections with a placeholder.
    """
    cleaned_results = []

    if "annotations" in task:
        for annotation in task["annotations"]:
            if "result" in annotation:
                new_results = []
                for r in annotation["result"]:
                    r_copy = copy.deepcopy(r)

                    # Skip unwanted result types outright
                    if r_copy.get("from_name") in keys_to_remove:
                        continue

                    # Scrub keys from the value dict if present
                    if "value" in r_copy and isinstance(r_copy["value"], dict):
                        for k in list(r_copy["value"].keys()):
                            if k in keys_to_remove:
                                del r_copy["value"][k]

                    # Replace label values
                    if r_copy.get("type") == "labels" and "value" in r_copy and "labels" in r_copy["value"]:
                        r_copy["value"]["labels"] = ["!!CHOOSE NEW LABEL!!!"]

                    new_results.append(r_copy)

                cleaned_results.extend(new_results)

    # Store cleaned results in predictions, keep annotations intact
    if cleaned_results:
        task["predictions"] = [{
            "model_version": "preannotation_v1",
            "result": cleaned_results
        }]

    return task


def _should_ignore_task(task, min_task_id=None, ignore_ids=None):
    """
    Return True if the task should be ignored based on:
    - min_task_id: ignore if numeric id < min_task_id
    - ignore_ids: ignore if task['id'] or task['__id__'] is in this set/list
    """
    ignore_ids = set(ignore_ids or [])

    # Match against both integer id and string __id__
    tid = task.get("id", None)
    sid = task.get("__id__", None)

    # If ignore list provided, check both forms
    if tid in ignore_ids or sid in ignore_ids:
        return True

    # Enforce min_task_id if we can interpret tid as an int
    if min_task_id is not None and tid is not None:
        try:
            if int(tid) < int(min_task_id):
                return True
        except (ValueError, TypeError):
            # If id isn't numeric, we can't apply min_task_id — just don't filter by it
            pass

    return False


def filter_first_x_tasks_single_annotator(tasks, x=10, min_task_id=None, ignore_ids=None):
    """
    Keep only the first X tasks per annotator, **only** if each task has exactly one annotator,
    and after filtering out tasks that:
      - have id < min_task_id (if provided and task['id'] is numeric), or
      - appear in ignore_ids (match against task['id'] or task['__id__']).

    Note: order of 'first' is the order of 'tasks' given to this function.
    """
    annotator_counts = defaultdict(int)
    filtered_tasks = []

    for task in tasks:
        # Apply additional filters first
        if _should_ignore_task(task, min_task_id=min_task_id, ignore_ids=ignore_ids):
            continue

        if "annotations" in task:
            annotator_ids = {a["completed_by"]["id"] for a in task["annotations"] if "completed_by" in a}

            # Skip tasks that have more than one annotator
            if len(annotator_ids) != 1:
                if annotator_ids:
                    print("MUltiple ANNOTATORS:", len(annotator_ids), annotator_ids)
                    print(task)
                continue

            annotator_id = next(iter(annotator_ids))  # Extract the single annotator
            if annotator_counts[annotator_id] < x:
                annotator_counts[annotator_id] += 1
                filtered_tasks.append(task)

    return filtered_tasks


def main(input_file_path, output_file_path, x=20, min_task_id=None, ignore_ids=None):
    """
    Load, filter (single annotator + id filters), create predictions, and save JSON file.

    Parameters:
      - x: max tasks per annotator to keep
      - min_task_id: ignore tasks with numeric id < this value
      - ignore_ids: iterable of ids to skip (matches task['id'] or task['__id__'])
    """
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter: only single annotator tasks + first X per annotator,
    # after applying min_task_id and ignore_ids filters
    filtered_data = filter_first_x_tasks_single_annotator(
        data, x=x, min_task_id=min_task_id, ignore_ids=ignore_ids
    )

    # Create cleaned predictions but keep original annotations untouched
    processed_data = [create_cleaned_predictions(task) for task in filtered_data]

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Processed file saved to {output_file_path}")


if __name__ == "__main__":
    input_path = "../annotationData/words/8_4_annotation_data.json"
    output_path = "../processed_data/cleaned_word_export_first20.json"

    # Example usage:
    # - keep first 20 per annotator
    # - ignore tasks with id < 5000
    # - and skip a few explicit ids (both numeric 'id' and string '__id__' supported)
    main(
        input_file_path=input_path,
        output_file_path=output_path,
        x=20,
        min_task_id=5000,                 # or None if not needed
        ignore_ids=None  # [] / None if not needed
    )
