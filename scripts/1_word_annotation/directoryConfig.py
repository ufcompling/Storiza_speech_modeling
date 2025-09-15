# ---- Paths (edit these if your storage layout changes) ----
import os
from pathlib import Path
ONEDRIVE_DIR = Path("~/University of Florida/").expanduser() 
PROJECT_SUBDIR = "Leite,Walter - Storiza Corpus Spring 2025"
UTTERANCES_SUBDIR = "Individual utterances"
SESSION_SUBDIR = "Storiza_Participant_Recording_05-30-25"

# Base folders
BUFFERED_UTTERANCES_SUBDIR = os.path.join(ONEDRIVE_DIR, PROJECT_SUBDIR, UTTERANCES_SUBDIR, "audio_clips")
AUDIO_BASE_DIR = os.path.join(ONEDRIVE_DIR, PROJECT_SUBDIR, SESSION_SUBDIR)
#PROCESSED_UTTERANCE_DIR = os.path.join(ONEDRIVE_DIR, PROJECT_SUBDIR,"Processed_Utterances")
PROCESSED_UTTERANCE_DIR = "processed_audio/"
ROOT_DIR = os.path.join("..", "..")  # repo root (relative to this script)
#PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_annotations")
#FIXES_DIR = os.path.join(ROOT_DIR, "fixes")
PROCESSED_DIR = "processed_annotations/"
FIXES_DIR = "fixes/"

# Inputs
# Cross-annotation check input (optional block at bottom)
CROSS_ANNOT_CSV = "export_157618_project-157618-at-2025-08-04-03-01-f5029ec4.csv"

JSON_FILENAME = "export_157618_project-157618-at-2025-09-15-19-14-9110b450.json"
JSON_PATH = os.path.join(PROCESSED_DIR, JSON_FILENAME)
SENTENCE_LABELS_CSV = os.path.join(PROCESSED_DIR, "sentenceLabels_with_comments.csv")

#--------------- Don't Change -------------------------------

# Outputs
STORY_LEVEL_CSV = os.path.join(PROCESSED_DIR, "story_level_data.csv")
SENTENCE_LEVEL_CSV = os.path.join(PROCESSED_DIR, "sentence_level_data.csv")
WORD_LEVEL_CSV = os.path.join(PROCESSED_DIR, "word_level_data.csv")
WORD_LEVEL_NGRAM_CSV = os.path.join(PROCESSED_DIR, "word_level_data_ngram.csv")
#FORMATTED_FULL_CSV = os.path.join("..", "..", "processed_annotations", "formatted_annotations_with_comments.csv")
FORMATTED_FULL_CSV = os.path.join("processed_annotations", "formatted_annotations_with_comments.csv")

# Audio slice output folders (written alongside the source audio session folder)
SENTENCE_SEGMENTS_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_sentence_segments")
WORD_SEGMENTS_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_word_segments")
WORD_SEGMENTS_NGRAM_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_word_segments_ngram")

