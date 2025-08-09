import os
import json
import pandas as pd
import re

# Paths
audio_dir = "../raw_data/audio"
xlsx_path = "../raw_data/Filtered_Story_Data_-_Missing_Files_Only.csv"
base_url = "https://2025storiza.michaelbennie.org/"

# Read Excel data
df = pd.read_csv(xlsx_path)

# Prepare output list
label_studio_tasks = []

# List all wav files in the audio directory
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Process each audio file
for audio_file in audio_files:
    # Extract userId and __id__ from filename
    match = re.match(r"uid_(.*?)_sid_(.*?)_(.*?)\.wav", audio_file)
    if not match:
        continue
    user_id, audio_id, _ = match.groups()

    # Find matching row in Excel (column names assumed from your description)
    row = df[(df['userId (matches the uid in the recording file name)'] == user_id) & (df['__id__'] == audio_id)]
    if row.empty:
        continue
    row = row.iloc[0]

    # Extract and process text content
    content = row['content']
    sentences = re.split(r'([.?!]["\']?\s+)(?=[A-Z])', content.strip())

    # Recombine the split pattern chunks
    combined = []
    for i in range(0, len(sentences), 2):
        chunk = sentences[i]
        if i + 1 < len(sentences):
            chunk += sentences[i + 1]
        combined.append(chunk.strip())

    sentences = [s for s in combined if s]
    # Format gold standard text
    gold_standard_text = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(sentences)])

    # Create possibleSentences list
    possible_sentences = [{"value": s} for s in sentences]
    possible_sentences.append({"value": "Other", "hint":"The child did not produce any of the above sentences. If there are only minor differences between the produced and target sentence, do not choose this option."})

    # Create task entry
    task = {
        "data": {
            "audio": base_url + audio_file,
            "goldStandardText": gold_standard_text,
            "possibleSentences": possible_sentences,
            "grade": row.get("grade", ""),
            "sound": row.get("sound", ""),
            "title": row.get("title", ""),
            "topic": row.get("topic", ""),
            "words": row.get("words", ""),
            "__id__": row.get("__id__", ""),
            "content": row.get("content", ""),
            "time": row.get("time", ""),
            "picture": row.get("picture", ""),
            "userId (matches the uid in the recording file name)": row.get(
                "userId (matches the uid in the recording file name)", ""),
            "matching_file": row.get("matching_file", "")
        }
    }
    if type(row.get("matching_file"))!= str:
        print("AAAAAA")
    else:
        label_studio_tasks.append(task)

# Save to JSON
output_path = "../processed_data/label_studio_audio_tasks.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(label_studio_tasks, f, indent=2)

print(f"Saved {len(label_studio_tasks)} tasks to {output_path}")
