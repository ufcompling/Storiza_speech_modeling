import pandas as pd


## Looking at the JSON format
def print_structure_with_values(d, indent=0, max_string_length=80):
    prefix = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}- {k}:")
                print_structure_with_values(v, indent + 1)
            else:
                val_str = str(v)
                if len(val_str) > max_string_length:
                    val_str = val_str[:max_string_length] + "..."
                print(f"{prefix}- {k}: {val_str}")
    elif isinstance(d, list):
        if not d:
            print(f"{prefix}- []")
        else:
            for i, item in enumerate(d):
                if isinstance(item, (dict, list)):
                    print_structure_with_values(item, indent)
                else:
                    val_str = str(item)
                    if len(val_str) > max_string_length:
                        val_str = val_str[:max_string_length] + "..."
                    print(f"{prefix}- [{i}]: {val_str}")



def get_annotation_type_sets(data):
    """
    Extract unique 'type', 'from_name', and (type, from_name) pairs
    from the first annotation block of each task in a Label Studio JSON export.

    Parameters
    ----------
    data : list[dict]
        Parsed JSON list where each item has an 'annotations' list.

    Returns
    -------
    tuple[set, set, set]
        (types, from_names, type_from_name_pairs)
    """
    # Example usage:
    # types, from_names, pairs = get_annotation_type_sets(data)
    # print(types); print(from_names); print(pairs)

    types = set()
    from_names = set()
    type_from_name_pairs = set()

    for tok in data:
        annotations = tok.get('annotations') or []
        if not annotations:
            continue
        results = annotations[0].get('result') or []
        for result in results:
            r_type = result.get('type')
            from_name = result.get('from_name')

            if r_type is not None:
                types.add(r_type)
            if from_name is not None:
                from_names.add(from_name)
            # Keep the pair even if one side is None to reflect what's present
            type_from_name_pairs.add((r_type, from_name))

    return types, from_names, type_from_name_pairs





def load_sentence_label_timestamps(sentence_labels_csv):
    """
    Load sentence-level labels and build a dictionary mapping
    (original_audio + goldStandard + repeated) → [start_time, end_time].

    Parameters
    ----------
    sentence_labels_csv : str
        Path to sentenceLabels_with_comments.csv

    Returns
    -------
    dict
        { identifier: [start_time, end_time] }
        where identifier = "<original_audio> <goldStandard> <repeated>"
    """
    sentenceLabels_data = pd.read_csv(sentence_labels_csv)

    annotated_intended_sentences_list = sentenceLabels_data['goldStandard'].tolist()
    original_audio_list = sentenceLabels_data['original_audio_name'].tolist()
    start_time_list = sentenceLabels_data['start_time'].tolist()
    end_time_list = sentenceLabels_data['end_time'].tolist()
    repeated_list = sentenceLabels_data['repeated'].tolist()

    sentenceLabels_original_audio_timestamps = {}
    for i, original_audio in enumerate(original_audio_list):
        if pd.isna(original_audio):
            continue

        start_time = start_time_list[i]
        end_time = end_time_list[i]
        repeated = 'true' if repeated_list[i] else 'false'
        goldStandard = annotated_intended_sentences_list[i]

        # Special case cleanup
        if 'Want to leap and play' in goldStandard:
            goldStandard = "\"Hi, Dawn!\" hooted Dale,\"Want to leap and play?\""
        if "Nate shouted, his voice a booming echo in the forest" in goldStandard:
            goldStandard = "You won't escape me, Jake! Nate shouted, his voice a booming echo in the forest."

        identifier = f"{original_audio} {goldStandard} {repeated}"
        sentenceLabels_original_audio_timestamps[identifier] = [start_time, end_time]

    return sentenceLabels_original_audio_timestamps