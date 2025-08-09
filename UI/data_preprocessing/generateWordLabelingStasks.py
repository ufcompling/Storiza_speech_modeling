import pydub
import os
from tqdm import tqdm
import pandas as pd
import json

from data_preprocessing.IPADict import IpaDictionary


def generate_audio_segment(audio_file_path, start_time, end_time, padding=1, output_directory='../processed_data/audio_clips/'):
    """
    Generates a segment of the audio file from start_time to end_time with padding.
    Saves it as an MP3 file in the specified output directory.
    """
    # Load the audio file
    audio = pydub.AudioSegment.from_wav(audio_file_path)

    # Get the total length of the audio file (in milliseconds)
    audio_length_ms = len(audio)

    # Convert start and end times to milliseconds with padding
    start_ms = max((start_time - padding), 0) * 1000  # Ensure start time does not go below 0
    end_ms = min((end_time + padding) * 1000, audio_length_ms)  # Ensure end time does not exceed the audio length

    # Extract the segment
    audio_segment = audio[start_ms:end_ms]

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate file name in the required format
    audio_name = os.path.basename(audio_file_path).split('.')[0]  # Strip file extension
    output_file = os.path.join(output_directory, f"{start_time:.1f}_end_{end_time:.1f}_{audio_name}.mp3")

    # Export the segment as an MP3
    audio_segment.export(output_file, format='mp3')

    return output_file


def safe_str(v):
    """Return v unchanged if it’s already a str; else '' if null/NaN; else str(v)."""
    if isinstance(v, str):
        return v
    if pd.isna(v):          # catches NaN, None, pd.NA, etc.
        return ""
    return str(v)


def generate_json_from_tsv(tsv_file_path, audio_input_directory,audio_output_directory):
    """
    Reads a TSV file, extracts relevant data, and generates a JSON object with audio metadata and annotations.
    """
    df = pd.read_csv(tsv_file_path, sep='\t')
    json_data = []
    ipa_dict = IpaDictionary('../raw_data/EnglishData.tsv')
    # Process each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Prepare data for the JSON entry
        audio_name = row['audio']
        audio_file_path = os.path.join(audio_input_directory, audio_name)
        segment_path = generate_audio_segment(audio_file_path, row['start_time'], row['end_time'])

        gold_standard = row['goldStandard'] if (row['goldStandard'] != "Other" or row["goldStandard"]is None) else ""

        actual_audio = row['actual'] if (row['actual'] is not None and type(row['actual'])==str) else gold_standard

        if type(gold_standard) != str or len(gold_standard) == 0:
            continue
            gold_standard = actual_audio
            actual_audio=""



        if(not row['nonchild']):
            segment_path = segment_path[len(audio_output_directory):]
            vocab = ipa_dict.get_vocab(gold_standard + " " + actual_audio)
            table_str = ipa_dict.format_string_table(vocab)
            data = {
                "data": {
                    # ---------- core -----------
                    "original_audio_name": audio_name,
                    "audio": f"https://2025storiza.michaelbennie.org/audio_clips/{segment_path}",
                    "goldStandard": gold_standard,
                    "actual": actual_audio,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "segment_time": row["segment_time"],  # end − start
                    "repeated": row["repeated"],
                    "runon": row["runon"],
                    "nonchild": row["nonchild"],
                    "IPAHints": table_str,

                    # ---------- extra metadata -----------
                    "annotator_id": row["annotator_id"],
                    "grade": safe_str(row["grade"]),
                    "sound": safe_str(row["sound"]),
                    "title": safe_str(row["title"]),
                    "topic": safe_str(row["topic"]),
                    "words": row["words"],
                    "__id__": row["__id__"],
                    "content": row["content"],
                    "time": row["time"],
                    "picture": row["picture"],
                    "userId": row["userId"],
                    "sentence_level_id":row["sentence_level_id"]
                }
            }

            json_data.append(data)

    # Convert data to JSON string
    json_output_path = os.path.join("./../processed_data/", 'audio_segments_data.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    return json_output_path


if __name__ == '__main__':
    generate_json_from_tsv('../processed_data/sentenceLabels.tsv', '../raw_data/audio/','../processed_data/audio_clips/')