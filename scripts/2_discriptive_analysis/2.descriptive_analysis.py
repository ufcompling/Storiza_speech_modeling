#!/usr/bin/env python
# coding: utf-8

# ---- Paths (edit these if your storage layout changes) ----
import os
from pathlib import Path

ONEDRIVE_DIR = Path("~/OneDrive/").expanduser()  # can also use "/Users/liu.ying/University of Florida/"
PROJECT_SUBDIR = "Leite,Walter's files - Storiza Corpus Spring 2025"
UTTERANCES_SUBDIR = "Individual utterances"
SESSION_SUBDIR = "Storiza_Participant_Recording_05-30-25"

# Base folders
BUFFERED_UTTERANCES_SUBDIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, UTTERANCES_SUBDIR, "audio_clips")
AUDIO_BASE_DIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, SESSION_SUBDIR)
PROCESSED_UTTERANCE_DIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, "Processed_Utterances")
ROOT_DIR = os.path.join("..", "..")  # repo root (relative to this script)
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_data")
FIXES_DIR = os.path.join(ROOT_DIR, "fixes")

# Inputs
# Cross-annotation check input (optional block at bottom)
CROSS_ANNOT_CSV = "export_157618_project-157618-at-2025-08-04-03-01-f5029ec4.csv"

JSON_FILENAME = "export_157618_project-157618-at-2025-09-01-21-16-2efa1216.json"
JSON_PATH = os.path.join(PROCESSED_DIR, JSON_FILENAME)
SENTENCE_LABELS_CSV = os.path.join(PROCESSED_DIR, "sentenceLabels_with_comments.csv")

#--------------- Don't Change -------------------------------

# Outputs
STORY_LEVEL_CSV = os.path.join(PROCESSED_DIR, "story_level_data.csv")
SENTENCE_LEVEL_CSV = os.path.join(PROCESSED_DIR, "sentence_level_data.csv")
WORD_LEVEL_CSV = os.path.join(PROCESSED_DIR, "word_level_data.csv")
WORD_LEVEL_NGRAM_CSV = os.path.join(PROCESSED_DIR, "word_level_data_ngram.csv")
FORMATTED_FULL_CSV = os.path.join("..", "..", "processed_data", "formatted_annotations_with_comments.csv")

# Audio slice output folders (written alongside the source audio session folder)
SENTENCE_SEGMENTS_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_sentence_segments")
WORD_SEGMENTS_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_word_segments")
WORD_SEGMENTS_NGRAM_DIR = os.path.join(PROCESSED_UTTERANCE_DIR, "annotated_word_segments_ngram")


# ======================= Imports ===========================
import numpy as np
import json
import pandas as pd
import librosa
from collections import defaultdict
import statistics
import ast


# =================== Descriptive analysis ==================
# (1) How many stories? Average production duration
# (2) How many gold-standard sentences on average per stories, average sentence length
# (3) How many actually produced sentences by children, on average per stories, average produced sentence length
# (4) How many parent sentences, repetitions, run-ons, etc
# (5) Error distributions
# (6) Word correct per minute

# --- Load sentence labels using configured path ---
sentenceLabels_data = pd.read_csv(SENTENCE_LABELS_CSV)

# Extract columns
temp_annotated_intended_sentences_list = sentenceLabels_data['goldStandard'].tolist()
temp_original_audio_list = sentenceLabels_data['original_audio_name'].tolist()
temp_sentenceLabels_start_time_list = sentenceLabels_data['start_time'].tolist()
temp_sentenceLabels_end_time_list = sentenceLabels_data['end_time'].tolist()
repeated_list = sentenceLabels_data['repeated'].tolist()
runon_list = sentenceLabels_data['runon'].tolist()
nonchild_list = sentenceLabels_data['nonchild'].tolist()

draft_ordered_sentences_list = sentenceLabels_data['ordered_sentences'].tolist()
temp_ordered_sentences_list = []

annotated_intended_sentences_list = []
original_audio_list = []
sentenceLabels_start_time_list = []
sentenceLabels_end_time_list = []

for i in range(len(nonchild_list)):
    nonchild = nonchild_list[i]
    if not nonchild:
        annotated_intended_sentences_list.append(temp_annotated_intended_sentences_list[i])
        original_audio_list.append(temp_original_audio_list[i])
        sentenceLabels_start_time_list.append(temp_sentenceLabels_start_time_list[i])
        sentenceLabels_end_time_list.append(temp_sentenceLabels_end_time_list[i])
        temp_ordered_sentences_list.append(draft_ordered_sentences_list[i])

# Counts for repeated, run-on, non-child
num_repeated = 0
num_runon = 0
num_nonchild = 0
for i in range(len(repeated_list)):
    repeated = repeated_list[i]
    runon = runon_list[i]
    nonchild = nonchild_list[i]
    if repeated:
        num_repeated += 1
    if runon:
        num_runon += 1
    if nonchild:
        num_nonchild += 1

print('Number of repeated sentences is', num_repeated)
print('Number of run-on sentences is', num_runon)
print('Number of non-child sentences is', num_nonchild)

# Stories/utterances
unique_original_audios = list(set(original_audio_list))
unique_original_audios = [audio for audio in unique_original_audios if not pd.isna(audio)]
num_stories = len(unique_original_audios)
print('Number of produced stories is', num_stories)
print('Number of produced utterances is', len(annotated_intended_sentences_list))

# Durations & sentence lists per story
utterance_duration_list = []
story_duration_dict = defaultdict(list)

ordered_sentences_list = []  # list of intended sentences from the original story (not the annotated intended sentences)
ordered_sentences_dict = {}

for audio in unique_original_audios:
    if audio not in story_duration_dict:
        story_duration_dict[audio] = []
        ordered_sentences_dict[audio] = []
        for i in range(len(sentenceLabels_start_time_list)):
            ordered_sentences = temp_ordered_sentences_list[i]
            try:
                ordered_sentences = ast.literal_eval(ordered_sentences)
                if ordered_sentences not in ordered_sentences_list:
                    ordered_sentences_list.append(ordered_sentences)
            except Exception:
                pass

            # times
            start_time = float(sentenceLabels_start_time_list[i])
            end_time = float(sentenceLabels_end_time_list[i])
            duration = end_time - start_time
            utterance_duration_list.append(duration)

            if original_audio_list[i] == audio:
                story_duration_dict[audio].append(duration)
                # Keep the most recent parsed ordered_sentences seen for this audio
                if isinstance(ordered_sentences, list):
                    ordered_sentences_dict[audio] = ordered_sentences

# Print ordered sentence mapping (optional)
for k, v in ordered_sentences_dict.items():
    print(k, v)

# Aggregate story duration stats
if num_stories > 0:
    ave_story_duration = sum(utterance_duration_list) / num_stories
    if ave_story_duration < 60:
        ave_story_duration = round(ave_story_duration, 2)
        print('Average story duration is', ave_story_duration, 'seconds')
    else:
        ave_story_min = round(ave_story_duration // 60)
        ave_story_secs = round(ave_story_duration % 60)
        print('Average story duration is', f'{ave_story_min}min{ave_story_secs}s')

story_duration_list = [sum(v) for v in story_duration_dict.values() if v]

if story_duration_list:
    shortest_story_duration = min(story_duration_list)
    if shortest_story_duration < 60:
        shortest_story_duration = round(shortest_story_duration, 2)
        print('Shortest story duration is', shortest_story_duration, 'seconds')
    else:
        shortest_story_min = round(shortest_story_duration // 60)
        shortest_story_secs = round(shortest_story_duration % 60)
        print('Shortest story duration is', f'{shortest_story_min}min{shortest_story_secs}s')

    longest_story_duration = max(story_duration_list)
    if longest_story_duration < 60:
        longest_story_duration = round(longest_story_duration, 2)
        print('Longest story duration is', longest_story_duration, 'seconds')
    else:
        longest_story_min = round(longest_story_duration // 60)
        longest_story_secs = round(longest_story_duration % 60)
        print('Longest story duration is', f'{longest_story_min}min{longest_story_secs}s')

# Utterance duration stats
if utterance_duration_list:
    ave_utterance_duration = statistics.mean(utterance_duration_list)
    if ave_utterance_duration < 60:
        ave_utterance_duration = round(ave_utterance_duration, 2)
        print('Average utterance duration is', ave_utterance_duration, 'seconds')
    else:
        ave_utterance_min = round(ave_utterance_duration // 60)
        ave_utterance_secs = round(ave_utterance_duration % 60)
        print('Average utterance duration is', f'{ave_utterance_min}min{ave_utterance_secs}s')

    shortest_utterance_duration = min(utterance_duration_list)
    if shortest_utterance_duration < 60:
        shortest_utterance_duration = round(shortest_utterance_duration, 2)
        print('Shortest utterance duration is', shortest_utterance_duration, 'seconds')
    else:
        shortest_utterance_min = round(shortest_utterance_duration // 60)
        shortest_utterance_secs = round(shortest_utterance_duration % 60)
        print('Shortest utterance duration is', f'{shortest_utterance_min}min{shortest_utterance_secs}s')

    longest_utterance_duration = max(utterance_duration_list)
    if longest_utterance_duration < 60:
        longest_utterance_duration = round(longest_utterance_duration, 2)
        print('Longest utterance duration is', longest_utterance_duration, 'seconds')
    else:
        longest_utterance_min = round(longest_utterance_duration // 60)
        longest_utterance_secs = round(longest_utterance_duration % 60)
        print('Longest utterance duration is', f'{longest_utterance_min}min{longest_utterance_secs}s')

# Count total ordered sentences across all stories
num_ordered_sentences = 0
for sent_list in ordered_sentences_list:
    num_ordered_sentences += len(sent_list)
print('Number of ordered intended sentences from the original stories is', num_ordered_sentences)

# Group annotated intended sentences by original audio names
audio_annotated_intended_sentences = defaultdict(list)
for audio in unique_original_audios:
    audio_annotated_intended_sentences[audio] = []
    for i in range(len(annotated_intended_sentences_list)):
        if original_audio_list[i] == audio:
            sentence = annotated_intended_sentences_list[i]
            audio_annotated_intended_sentences[audio].append(sentence)

# Identify mismatches between annotated vs original intended sentence counts
for audio in unique_original_audios:
    annotated_intended_sentences = audio_annotated_intended_sentences[audio]
    original_intended_sentences = ordered_sentences_dict.get(audio, [])
    if len(annotated_intended_sentences) != len(original_intended_sentences):
        print(annotated_intended_sentences)
        print(original_intended_sentences)
        print('\n')

# Average sentence lengths
annotated_intended_sent_len_list = []
for _, v in audio_annotated_intended_sentences.items():
    for sent in v:
        if isinstance(sent, str):
            sent_len = len(sent.split())
            annotated_intended_sent_len_list.append(sent_len)

if annotated_intended_sent_len_list:
    print('Number of annotated intended sentences is', len(annotated_intended_sent_len_list))
    print('Average sentence length for annotated intended sentences is', round(statistics.mean(annotated_intended_sent_len_list)))

original_intended_sent_len_list = []
for sent_list in ordered_sentences_list:
    for sent in sent_list:
        if isinstance(sent, str):
            sent_len = len(sent.split())
            original_intended_sent_len_list.append(sent_len)

if original_intended_sent_len_list:
    print('Number of original intended sentences is', len(original_intended_sent_len_list))
    print('Average sentence length for original intended sentences is', round(statistics.mean(original_intended_sent_len_list)))

# ================= Word correct per minute (WCPM) =================
# Load using configured path
word_segments_data = pd.read_csv(WORD_LEVEL_CSV)
word_path_list = word_segments_data['Path'].tolist()
word_error_categories_list = word_segments_data['Error Category'].tolist()
word_error_labels_list = word_segments_data['Error Labels'].tolist()

story_error_categories_dict = {}
story_error_labels_dict = {}
for audio in unique_original_audios:
    story_error_categories_dict[audio] = []
    story_error_labels_dict[audio] = []
    audio_name = os.path.splitext(audio)[0]
    for i in range(len(word_path_list)):
        # Use basename for portability (avoids hardcoding '/')
        path = os.path.basename(word_path_list[i])
        if path.startswith(audio_name):
            error_categories = str(word_error_categories_list[i]).split('+')
            if error_categories and error_categories[0] == "Mixed Error":
                error_categories = error_categories[1:]
            for category in error_categories:
                category = category.strip()
                if category:
                    story_error_categories_dict[audio].append(category)
            try:
                error_labels = ast.literal_eval(word_error_labels_list[i])
                for label in error_labels:
                    story_error_categories_dict[audio].append(label)
            except Exception:
                pass
        else:
            pass

word_correct_per_min_list = []
for k, v in story_error_categories_dict.items():
    if v and story_duration_dict.get(k):
        story_duration = sum(story_duration_dict[k])
        num_word_correct = v.count('Correct')
        word_correct_per_min = (num_word_correct / story_duration) * 60 if story_duration > 0 else 0.0
        word_correct_per_min_list.append(word_correct_per_min)

if word_correct_per_min_list:
    print('Average word-correct-per-minute (WCPM):', round(statistics.mean(word_correct_per_min_list), 2))

# ================= Preprocessing for multi-label categories =================
word_segments_data_file = WORD_LEVEL_CSV
full_word_segments_data = pd.read_csv(word_segments_data_file)

# Keep all rows (optionally subsample with [:N] while debugging)
word_segments_data = full_word_segments_data  # [:8]

# Split multi-categories; drop "Mixed Error" marker
word_segments_data["Error Category"] = word_segments_data["Error Category"].apply(
    lambda x: [category.strip() for category in str(x).split("+") if category.strip() and category.strip() != 'Mixed Error']
)

# Transform Error Labels column into token lists
word_segments_data['Error Labels'] = word_segments_data['Error Labels'].apply(
    lambda labels: labels.strip('[]').split(', ') if isinstance(labels, str) and labels != '[]' else ['NONE']
)

# Prepare multi-hot encoding
all_error_categories = sorted({category for sublist in word_segments_data["Error Category"] for category in sublist})
num_error_categories = len(all_error_categories)
print('Error Categories:', all_error_categories)
print('Total number of unique error categories:', len(all_error_categories))

category2id = {category: i for i, category in enumerate(all_error_categories)}
id2category = {i: category for i, category in enumerate(all_error_categories)}

def encode_labels(error_category_list):
    vec = [0] * num_error_categories
    for category in error_category_list:
        if category in category2id:
            vec[category2id[category]] = 1
    return vec

word_segments_data['labels'] = word_segments_data['Error Category'].apply(encode_labels)

# Peek at the first few label vectors (optional)
print('Example multi-hot vectors:', word_segments_data['labels'].tolist()[:2])


# =================== Summary ===================
print("\n================ SUMMARY ==================")

# (1) How many stories? Average production duration
print(f"(1) Number of stories: {num_stories}")
if num_stories > 0:
    ave_story_duration = sum(utterance_duration_list) / num_stories
    print(f"    Average story duration: {round(ave_story_duration,2)} seconds "
          f"({round(ave_story_duration/60,2)} minutes)")

# (2) Gold-standard sentences
print(f"(2) Gold-standard sentences (intended): {len(original_intended_sent_len_list)} total")
if original_intended_sent_len_list:
    print(f"    Average sentence length (words): {round(statistics.mean(original_intended_sent_len_list),2)}")

# (3) Produced sentences by children
print(f"(3) Produced sentences (annotated): {len(annotated_intended_sent_len_list)} total")
if annotated_intended_sent_len_list:
    print(f"    Average sentence length (words): {round(statistics.mean(annotated_intended_sent_len_list),2)}")

# (4) Parent sentences, repetitions, run-ons
print(f"(4) Repeated: {num_repeated}, Run-on: {num_runon}, Non-child: {num_nonchild}")

# (5) Error distributions
print(f"(5) Error categories observed ({len(all_error_categories)} unique): {all_error_categories}")

# (6) Word correct per minute
if word_correct_per_min_list:
    print(f"(6) Average WCPM: {round(statistics.mean(word_correct_per_min_list),2)}")
else:
    print("(6) Average WCPM: N/A (no data)")

print("==========================================\n")
