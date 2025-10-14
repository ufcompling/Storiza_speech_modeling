#!/usr/bin/env python
# coding: utf-8

# ---- Paths (edit these if your storage layout changes) ----
import os
from pathlib import Path
from collections import defaultdict

ONEDRIVE_DIR = Path("~/University of Florida/").expanduser() 
PROJECT_SUBDIR = "Leite,Walter - Storiza Corpus Spring 2025"
UTTERANCES_SUBDIR = "Individual utterances"
SESSION_SUBDIR = "Storiza_Participant_Recording_05-30-25"

# Base folders
BUFFERED_UTTERANCES_SUBDIR = os.path.join(ONEDRIVE_DIR, PROJECT_SUBDIR, UTTERANCES_SUBDIR, "audio_clips")
AUDIO_BASE_DIR = os.path.join(ONEDRIVE_DIR, PROJECT_SUBDIR, SESSION_SUBDIR)
#BUFFERED_UTTERANCES_SUBDIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, UTTERANCES_SUBDIR, "audio_clips")
#AUDIO_BASE_DIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, SESSION_SUBDIR)
#PROCESSED_UTTERANCE_DIR = os.path.join(str(ONEDRIVE_DIR), PROJECT_SUBDIR, "Processed_Utterances")
PROCESSED_UTTERANCE_DIR = "processed_audio/"
ROOT_DIR = os.path.join("..", "..")  # repo root (relative to this script)
#PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_data")
#FIXES_DIR = os.path.join(ROOT_DIR, "fixes")
PROCESSED_DIR = "processed_annotations/"
FIXES_DIR = "fixes/"

# Inputs
# Cross-annotation check input (optional block at bottom)
CROSS_ANNOT_CSV = "export_157618_project-157618-at-2025-08-04-03-01-f5029ec4.csv"

JSON_FILENAME = "export_157618_project-157618-at-2025-10-12-18-32-ccdaca6d.json"
JSON_PATH = os.path.join(PROCESSED_DIR, JSON_FILENAME)
SENTENCE_LABELS_CSV = os.path.join(PROCESSED_DIR, "sentenceLabels_with_comments.csv")

#--------------- Don't Change -------------------------------

# Outputs
STORY_META_CSV = os.path.join(PROCESSED_DIR, "story.xlsx")
STORY_LEVEL_CSV = os.path.join(PROCESSED_DIR, "story_level_data.csv")
SENTENCE_LEVEL_CSV = os.path.join(PROCESSED_DIR, "sentence_level_data.csv")
WORD_LEVEL_CSV = os.path.join(PROCESSED_DIR, "word_level_data.csv")
WORD_LEVEL_NGRAM_CSV = os.path.join(PROCESSED_DIR, "word_level_data_ngram.csv")
FORMATTED_FULL_CSV = os.path.join("processed_annotations", "formatted_annotations_with_comments.csv")

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

STORY_META_CSV = os.path.join("processed_annotations/story.xlsx")
story_meta_data = pd.read_excel(STORY_META_CSV)
story_id_list = story_meta_data['__id__'].tolist()
grade_list = story_meta_data['grade'].tolist()
userID_list = story_meta_data['userId (matches the uid in the recording file name)'].tolist()
userID_grade_dict = {}
for i in range(len(userID_list)):
	userID = story_id_list[i] + ' ' + userID_list[i]
	if userID not in userID_grade_dict:
		userID_grade_dict[userID] = grade_list[i]

# --- Load sentence labels using configured path ---
sentenceLabels_data = pd.read_csv(SENTENCE_LABELS_CSV)

# Extract columns
repeated_list = sentenceLabels_data['repeated'].tolist()
runon_list = sentenceLabels_data['runon'].tolist()
nonchild_list = sentenceLabels_data['nonchild'].tolist()

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

sentenceLabels_annotated_intended_sentences_list = sentenceLabels_data['goldStandard'].tolist()
sentenceLabels_original_audio_list = sentenceLabels_data['original_audio_name'].tolist()
sentenceLabels_start_time_list = sentenceLabels_data['start_time'].tolist()
sentenceLabels_end_time_list = sentenceLabels_data['end_time'].tolist()
sentenceLabels_ordered_sentences_list = sentenceLabels_data['ordered_sentences'].tolist()

ordered_sentences_dict = defaultdict(list)
annotated_intended_sentences_dict = defaultdict(list)

# Extract information about actually produced utterances
sentence_level_data = pd.read_csv(SENTENCE_LEVEL_CSV)
produced_original_audio_list = sentence_level_data['original_audio_name'].tolist()
produced_start_time_list = sentence_level_data['Start Time'].tolist()
produced_end_time_list = sentence_level_data['End Time'].tolist()

# Not all children from story.xlsx have saved data
include_userID = []
for i in range(len(produced_original_audio_list)):
	original_audio_name = produced_original_audio_list[i]
	userID = original_audio_name.split('_')[1]
	storyID = original_audio_name.split('_')[3]
	include_userID.append(storyID + ' ' + userID)

exclude_userID = []
for k, v in userID_grade_dict.items():
	if k not in include_userID:
		exclude_userID.append(k)

for k in exclude_userID:
	del userID_grade_dict[k]

## Counting children and stories by different grade levels
children_list = []
kindergarten_count = 0
firstgrade_count = 0
secondgrade_count = 0
thirdgrade_count = 0

children_grade_dict = defaultdict(list)
for k, v in userID_grade_dict.items():
	child = k.split()[-1]
	if child not in children_list:
		children_list.append(child)
	if v not in children_grade_dict[child]:
		children_grade_dict[child].append(v)

num_children_multigrade = 0
for k, v in children_grade_dict.items():
	if len(v) > 1:
		num_children_multigrade += 1
print('\n')
print('Number of children with multiple grade levels:', num_children_multigrade)

kindergarten_count = 0
firstgrade_count = 0
secondgrade_count = 0
thirdgrade_count = 0

for k, v in userID_grade_dict.items():
	if v == 'Kindergarten':
		kindergarten_count += 1 
	if v == '1st Grade':
		firstgrade_count += 1 
	if v == '2nd Grade':
		secondgrade_count += 1 
	if v == '3rd Grade and Higher':
		thirdgrade_count += 1

print('Total number of children:', len(children_list))
print('Number of kindergartender:', kindergarten_count)
print('Number of 1st Grade:', firstgrade_count) 
print('Number of 2nd Grade:', secondgrade_count)
print('Number of 3rd Grade:', thirdgrade_count)

# Stories/utterances
produced_start_end_dict = defaultdict(list)
unique_original_audios = list(set(produced_original_audio_list))
num_stories = len(unique_original_audios)
print('\n')
print('Number of produced stories is', num_stories)

for audio in unique_original_audios:
	for i in range(len(produced_start_time_list)):
		original_audio_name = produced_original_audio_list[i]
		if original_audio_name == audio:
			start_time = float(produced_start_time_list[i])
			end_time = float(produced_end_time_list[i])
			produced_start_end_dict[original_audio_name].append([start_time, end_time])

for i in range(len(nonchild_list)):
	nonchild = nonchild_list[i]
	if not nonchild: # Filter out parent utterances
		original_audio_name = sentenceLabels_original_audio_list[i]
		if original_audio_name in produced_original_audio_list:
			if ast.literal_eval(sentenceLabels_ordered_sentences_list[i]) not in ordered_sentences_dict[original_audio_name]:
				ordered_sentences_dict[original_audio_name] = ast.literal_eval(sentenceLabels_ordered_sentences_list[i])
			annotated_intended_sentences_dict[original_audio_name].append(sentenceLabels_annotated_intended_sentences_list[i])


num_produced_utterances = 0
for k, v in annotated_intended_sentences_dict.items():
	num_produced_utterances += len(v)

print('\n')
print('Number of produced utterances is', num_produced_utterances)

num_ordered_sentences = 0
for k, v in ordered_sentences_dict.items():
	num_ordered_sentences += len(v)

print('\n')
print('Number of ordered intended sentences from the original stories is', num_ordered_sentences)


# Durations & sentence lists per story
utterance_duration_list = []
story_duration_dict = defaultdict(list)

for k, v in produced_start_end_dict.items():
	duration_sum = 0
	for start_end in v:
		duration = start_end[1] - start_end[0]
		utterance_duration_list.append(duration)
		duration_sum += duration
	story_duration_dict[k] = duration_sum


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

story_duration_list = [v for v in story_duration_dict.values()]

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

print('\n')

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

print('\n')

# Identify mismatches between annotated vs original intended sentence counts
num_story_mismatch_sent_count = 0
for audio in unique_original_audios:
	annotated_intended_sentences = annotated_intended_sentences_dict[audio]
	original_intended_sentences = ordered_sentences_dict.get(audio, [])
	if len(annotated_intended_sentences) != len(original_intended_sentences):
		num_story_mismatch_sent_count += 1
print('Number of stories with mismatches between annotated vs original intended sentence counts:', num_story_mismatch_sent_count)
print('\n')

# Average sentence lengths
original_intended_sent_len_list = []
for k, v in ordered_sentences_dict.items():
	for sent in v:
		if isinstance(sent, str):
			sent_len = len(sent.split())
			original_intended_sent_len_list.append(sent_len)

if original_intended_sent_len_list:
	print('Total number of original intended sentences is', len(original_intended_sent_len_list))
	print('Average sentence length for original intended sentences is', round(statistics.mean(original_intended_sent_len_list)))

annotated_intended_sent_len_list = []
for _, v in annotated_intended_sentences_dict.items():
	for sent in v:
		if isinstance(sent, str):
			sent_len = len(sent.split())
			annotated_intended_sent_len_list.append(sent_len)

if annotated_intended_sent_len_list:
	print('Total number of annotated intended sentences is', len(annotated_intended_sent_len_list))
	print('Average sentence length for annotated intended sentences is', round(statistics.mean(annotated_intended_sent_len_list)))

# ================= Word correct per minute (WCPM) =================
# Load using configured path
word_segments_data = pd.read_csv(WORD_LEVEL_CSV)
word_path_list = word_segments_data['Path'].tolist()

# Split multi-categories; drop "Mixed Error" marker
word_segments_data["Error Category"] = word_segments_data["Error Category"].apply(
	lambda x: [category.strip() for category in str(x).split("+") if category.strip() and category.strip() != 'Mixed Error']
)

word_segments_data['Error Labels'] = word_segments_data['Error Labels'].apply(lambda labels: labels.strip('[]').split(', ') if labels != '[]'  else ['NONE'])

word_error_categories_list = word_segments_data['Error Category'].tolist()
word_error_labels_list = word_segments_data['Error Labels'].tolist()
print(word_error_categories_list[:2])
print(word_error_labels_list[:2])

## Consolidate error categories
error_map = {
	'Grammatical': 'Grammatical Error',
	'Orthographic Sub.': 'Orthographic Error',
	'Phonological': 'Phonological Error',
	'Run-on': 'Run-on Word',
	'Structural': 'Structural Error',
	'Visual Tracking': 'Visual Tracking Error',
	'Contraction/Shortening': 'Correct'
}

modified_error_category_list = []
for i in range(len(word_error_categories_list)):
	error_categories = word_error_categories_list[i]
	for k, v in error_map.items():
		while k in error_categories:
			error_categories = [error_map[category] if category in error_map else category for category in error_categories]
	modified_error_category_list.append(error_categories)

## Getting error category distribution
error_category_dist = {}
for i in range(len(modified_error_category_list)):
	categories = modified_error_category_list[i]
	for category in categories:
		if category in error_category_dist:
			error_category_dist[category] += 1
		else:
			error_category_dist[category] = 1

sorted_error_category_dist = dict(sorted(error_category_dist.items(), key=lambda item: item[1], reverse=True))
print("Error Category Distribution")
for category, count in sorted_error_category_dist.items():
	print(f"{category}: {count} / {count/sum(sorted_error_category_dist.values()):.2%}")

## Getting error labels distribution
error_label_dist = {}
for i in range(len(word_error_labels_list)):
	labels = word_error_labels_list[i]
	for label in labels:
		if label in error_label_dist:
			error_label_dist[label] += 1
		else:
			error_label_dist[label] = 1

sorted_error_label_dist = dict(sorted(error_label_dist.items(), key=lambda item: item[1], reverse=True))

print('\n')
print("Error labels Distribution")
for label, count in sorted_error_label_dist.items():
	print(f"{label}: {count} / {count/sum(sorted_error_label_dist.values()):.2%}")


for grade_level in ['Kindergarten', '1st Grade', '2nd Grade', '3rd Grade and Higher']:
	print('WCPM for', grade_level)
	story_error_categories_dict = {}
	story_error_labels_dict = {}
	for audio in unique_original_audios:
		userID = audio.split('_')[1]
		storyID = audio.split('_')[3]
		grade = userID_grade_dict[storyID + ' ' + userID]
		if grade == grade_level:
			story_error_categories_dict[audio] = []
			story_error_labels_dict[audio] = []
			audio_name = os.path.splitext(audio)[0]
			for i in range(len(word_path_list)):
				# Use basename for portability (avoids hardcoding '/')
				path = os.path.basename(word_path_list[i])
				if path.startswith(audio_name):
					error_categories = word_error_categories_list[i]
					story_error_categories_dict[audio] += error_categories
					error_labels = word_error_labels_list[i]
					story_error_labels_dict[audio] += error_labels
				else:
					pass

	word_per_min_list = []
	word_correct_per_min_list = []
	for k, v in story_error_categories_dict.items():
		if v and story_duration_dict.get(k):
			for userID, grade in userID_grade_dict.items():
				if k.split('_')[3] + ' ' + k.split('_')[1] == userID: 
					story_duration = story_duration_dict[k]
					num_word = len(v)
					word_per_min = (num_word / story_duration) * 60 if story_duration > 0 else 0.0
					word_per_min_list.append(word_per_min)
					num_word_correct = v.count('Correct')
					word_correct_per_min = (num_word_correct / story_duration) * 60 if story_duration > 0 else 0.0
					word_correct_per_min_list.append(word_correct_per_min)

					if round(word_correct_per_min) == 185:
						print(k, v, 'LARGE WCPM', v.count('Correct'), story_duration)

	if word_per_min_list:
		word_per_min_list.sort()
		print('Average word-per-minute:', round(statistics.mean(word_per_min_list), 2))
		print('Lowest word-per-minute:', round(word_per_min_list[0]))
		print('Highest word-per-minute:', round(word_per_min_list[-1]))
	if word_correct_per_min_list:
		word_correct_per_min_list.sort()
		print('Average word-correct-per-minute (WCPM):', round(statistics.mean(word_correct_per_min_list), 2))
		print('Lowest word-correct-per-minute (WCPM):', round(word_correct_per_min_list[0]))
		print('Highest word-correct-per-minute (WCPM):', round(word_correct_per_min_list[-1]))

	print('\n')

# ================= Preprocessing for multi-label categories =================




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
print(f"(5) Error categories observed ({len(error_category_dist)} unique): {error_category_dist}")


print("==========================================\n")
