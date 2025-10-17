## Given predictions.txt of a pretrained model
## Evaluate its performance on just the test set, i.e., not the full sentence_level_data

## Also calculate oral reading fluency scored by the ASR
## for both the full sentence_level data as well as the test set

import requests
import torch
import os
import io
import sys
import soundfile as sf
import pandas as pd
import evaluate
import librosa

import string
import statistics

def strip_punctuation_except_apostrophe(text):
	"""
	Removes all punctuation from a string except for the apostrophe (').

	Args:
		text (str): The input string.

	Returns:
		str: The string with most punctuation removed, preserving apostrophes.
	"""
	# Get all punctuation characters
	all_punctuation = string.punctuation

	# Remove the apostrophe from the set of punctuation to be deleted
	punctuation_to_remove = all_punctuation.replace("'", "")

	# Create a translation table to delete the specified punctuation
	translator = str.maketrans('', '', punctuation_to_remove)

	# Apply the translation to the string
	cleaned_text = text.translate(translator)
	return cleaned_text

sentence_level_data = pd.read_csv('processed_annotations/sentence_level_data.csv')
audio_list = sentence_level_data['Path'].tolist()
original_transcript_list = sentence_level_data['Transcript'].tolist()
transcript_list = [strip_punctuation_except_apostrophe(transc) for transc in original_transcript_list]
transcript_list = [transc.lower() for transc in transcript_list]
goldStandard_list = sentence_level_data['goldStandard'].tolist()

# Audio: duration
audio_duration_dict = {}
for i in range(len(audio_list)):
	audio = audio_list[i]
	duration = librosa.get_duration(path=audio)
	audio_duration_dict[audio] = duration

# Audio: goldStandard utterance
goldStandard_dict = {}
for i in range(len(audio_list)):
	goldStandard_dict[audio_list[i]] = goldStandard_list[i]

# Audio: original manual transcript
gold_original_dict = {}
for i in range(len(audio_list)):
	gold_original_dict[audio_list[i]] = original_transcript_list[i]

# Audio: manual transcript after stripping punctuations and lowercase
gold_strip_dict = {}
for i in range(len(audio_list)):
	gold_original_dict[audio_list[i]] = transcript_list[i]

# Getting predictions
prediction_path = sys.argv[1]
prediction_list = []
with open(prediction_path) as f:
	for line in f:
		prediction_list.append(line.strip())

assert len(audio_list) == len(prediction_list)

prediction_dict = {}
for i in range(len(audio_list)):
	prediction_dict[audio_list[i]] = prediction_list[i]

# Getting test data
test_sentence_level_data = pd.read_csv('asr_random/sentence_test_1.csv')
test_audio_list = sentence_level_data['Path'].tolist()
test_original_transcript_list = sentence_level_data['Transcript'].tolist()
test_transcript_list = [strip_punctuation_except_apostrophe(transc) for transc in original_transcript_list]
test_transcript_list = [transc.lower() for transc in transcript_list]

test_original_transcript_list = []
test_strip_transcript_list = []
test_original_pred_list = []
test_strip_pred_list = []
for i in range(len(test_audio_list)):
	audio = test_audio_list[i]
	test_original_transcript_list.append(gold_original_dict[audio])
	test_strip_transcript_list.append(gold_strip_dict[audio])
	test_original_pred_list.append(prediction_dict[audio])
	test_strip_pred_list.append(strip_punctuation_except_apostrophe(prediction_dict[audio]).lower())

## Evaluating pretrained ASR model on test set
original_wer = wer_metric.compute(predictions=test_original_pred_list, references=test_original_transcript_list)
original_cer = cer_metric.compute(predictions=test_original_pred_list, references=test_original_transcript_list)

strip_wer = wer_metric.compute(predictions=test_strip_pred_list, references=test_strip_transcript_list)
strip_cer = cer_metric.compute(predictions=test_strip_pred_list, references=test_strip_transcript_list)
print('ASR evaluation of pretrained model on the test set')
print('Original WER:', original_wer)
print('Original CER:', original_cer)
print('\n')
print('Strip WER:', strip_wer)
print('Strip CER:', strip_cer)

## Automated oral reading fluency scoring on the full sentence_level_data
full_num_correct_by_sentence = []
for audio, duration in audio_duration_dict.items():
	num_correct = 0
	goldStandard = goldStandard_dict[audio]
	goldStandard = strip_punctuation_except_apostrophe(goldStandard).lower()
	pred = prediction_dict[audio]
	pred = strip_punctuation_except_apostrophe(pred).lower()
	for w in goldStandard:
		if w in pred:
			num_correct += 1
	wcpm = (num_correct / duration) * 60.0
	full_num_correct_by_sentence.append(wcpm)


print('Automated oral reading fluency scoring on the full sentence_level_data')
print('WCPM:', statistics.mean(full_num_correct_by_sentence))
print('\n')

## Automated oral reading fluency scoring on the test set
test_num_correct_by_sentence = []
for audio in test_audio_list:
	num_correct = 0
	duration = audio_duration_dict[audio]
	goldStandard = goldStandard_dict[audio]
	goldStandard = strip_punctuation_except_apostrophe(goldStandard).lower()
	pred = prediction_dict[audio]
	pred = strip_punctuation_except_apostrophe(pred).lower()
	for w in goldStandard:
		if w in pred:
			num_correct += 1
	wcpm = (num_correct / duration) * 60.0
	test_num_correct_by_sentence.append(wcpm)

print('Automated oral reading fluency scoring on the test set')
print('WCPM:', statistics.mean(test_num_correct_by_sentence))
print('\n')

#output_path = sys.argv[2]
