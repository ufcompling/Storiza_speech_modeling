import json
from collections import defaultdict
import copy

# Keys to remove from annotations
keys_to_remove = {
    "DisfluencyErrorType", "StructuralErrorType", "GrammaticalErrorType",
    "OrthographicErrorType", "PhonologicalErrorType", "VisualTrackingErrorType",
    "MixedErrorTaxonomy", "mispronunciation_word", "produced_word",
    "spoken_word", "spoken_words"
}


def create_cleaned_predictions(task):
    """Create cleaned predictions from annotations without modifying original annotations."""
    cleaned_results = []

    if "annotations" in task:
        for annotation in task["annotations"]:
            if "result" in annotation:
                # Deep copy to avoid modifying original annotations
                new_results = []
                for r in annotation["result"]:
                    r_copy = copy.deepcopy(r)

                    # Skip unwanted types
                    if r_copy.get("from_name") in keys_to_remove:
                        continue

                    # Replace label values
                    if r_copy.get("type") == "labels" and "value" in r_copy:
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


def filter_first_x_tasks_single_annotator(tasks, x=10):
    """Keep only the first X tasks per annotator, only if task has exactly one annotator."""
    annotator_counts = defaultdict(int)
    filtered_tasks = []

    for task in tasks:
        if "annotations" in task:
            annotator_ids = {a["completed_by"]["id"] for a in task["annotations"] if "completed_by" in a}

            # Skip tasks that have more than one annotator
            if len(annotator_ids) != 1:
                if annotator_ids:
                    print("MUltiple ANNOTATORS:", len(annotator_ids),annotator_ids)
                    print(task)
                continue

            annotator_id = next(iter(annotator_ids))  # Extract the single annotator
            if annotator_counts[annotator_id] < x:
                annotator_counts[annotator_id] += 1
                filtered_tasks.append(task)

    return filtered_tasks


def main(input_file_path, output_file_path):
    """Load, filter (single annotator), create predictions, and save JSON file."""
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter: only single annotator tasks + first X per annotator
    filtered_data = filter_first_x_tasks_single_annotator(data, x=20)

    # Create cleaned predictions but keep original annotations untouched
    processed_data = [create_cleaned_predictions(task) for task in filtered_data]

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Processed file saved to {output_file_path}")


if __name__ == "__main__":
    input_path = "../annotationData/words/8_4_annotation_data.json"
    output_path = "../processed_data/cleaned_word_export_first20.json"
    main(input_path, output_path)
