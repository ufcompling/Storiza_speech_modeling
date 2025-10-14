# This scripts directly evaluate wav2vec2-large-960h on the audio data

import numpy as np
import re, os
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
import evaluate
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import soundfile as sf
import argparse
from tqdm import tqdm
from random import shuffle
import pandas as pd
import librosa
import shutil
from transformers import AutoModelForCTC, AutoProcessor
from pyctcdecode import build_ctcdecoder 

if not os.path.exists('asr_model/'):
	os.system('mkdir asr_model/')

def read_audio(fname):
	""" Load an audio file and return PCM along with the sample rate """
	wav, sr = sf.read(fname)
	return wav, sr

chars_to_ignore_regex = '[\,\?\!\-\;\"\(\)\&\-\>\[\]\_\ˈ]'

def clean_sent(transcript):
	while '/' in transcript:
		transcript = transcript.replace('/', '')
	while '|' in transcript:
		transcript = transcript.replace('|', '/')
	while '1' in transcript:
		transcript = transcript.replace('1', 'one')
	while '2' in transcript:
		transcript = transcript.replace('2', 'two')
	while '3' in transcript:
		transcript = transcript.replace('3', 'three')
	transcript = re.sub(chars_to_ignore_regex, '', transcript).lower().strip()
	return transcript

def get_data_simple(data_path, file):
	"""Load data without LM building"""
	data_path_file = data_path + file
	data_path_file = data_path_file.replace(u'\xa0', u'')
	original_data = pd.read_csv(data_path_file)
	path_list = original_data['path'].tolist()
	transcript_list = original_data['transcript'].tolist()

	## Getting audio data
	data = []
	duration = 0

	for i in range(len(path_list)):
		wav_path = path_list[i]
		try:
			transcript = clean_sent(transcript_list[i])
			signal, samplerate = librosa.load(wav_path, sr=16000)
			
			if len(signal.shape) > 1 and signal.shape[1] == 2:
				signal = np.average(signal, axis=1)

			entry = {}
			duration += len(signal) / samplerate
			entry["sentence"] = transcript.replace("\n", " ")				
			entry["audio"] = {"sampling_rate" : samplerate, "array" : signal}
			data.append(entry)
		except Exception as e:
			print(f"Error processing {wav_path}: {e}")
			continue

	print(f"Total duration: {duration:.2f} seconds")
	print(f"Loaded {len(data)} samples")

	return data

def evaluate_pretrained_model(pretrained_model_name, test_data, output_dir, batch_size=8):
	"""
	Evaluate a pretrained model with batch processing
	"""
	print("\n" + "="*50)
	print(f"EVALUATING PRETRAINED MODEL: {pretrained_model_name}")
	print("="*50)
	
	os.makedirs(output_dir, exist_ok=True)
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	
	# Load model
	from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
	
	try:
		processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
		model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name).to(device)
		print("✓ Successfully loaded model and processor")
	except Exception as e:
		print(f"✗ Error loading model: {e}")
		return None, None
	
	# Extract audio and sentences
	audios = []
	sentences = []
	
	for item in test_data:
		audio = item["audio"]["array"]
		
		# Handle stereo
		if len(audio.shape) > 1:
			audio = np.mean(audio, axis=1 if audio.shape[1] == 2 else 0)
		
		# Ensure 1D and correct dtype
		audio = audio.flatten().astype(np.float32)
		
		audios.append(audio)
		sentences.append(item["sentence"])
	
	print(f"Loaded {len(audios)} audio samples")
	
	# Process in batches
	predictions = []
	model.eval()
	
	for batch_start in tqdm(range(0, len(audios), batch_size)):
		batch_end = min(batch_start + batch_size, len(audios))
		batch_audios = audios[batch_start:batch_end]
		
		try:
			# Process batch
			inputs = processor(
				batch_audios,
				sampling_rate=16000,
				return_tensors="pt",
				padding=True
			)
			
			input_values = inputs.input_values.to(device)
			attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
			
			# Inference
			with torch.no_grad():
				if attention_mask is not None:
					logits = model(input_values, attention_mask=attention_mask).logits
				else:
					logits = model(input_values).logits
			
			# Decode
			pred_ids = torch.argmax(logits, dim=-1)
			batch_preds = processor.batch_decode(pred_ids)
			
			predictions.extend([p.lower() for p in batch_preds])
			
		except Exception as e:
			print(f"\nError processing batch {batch_start}-{batch_end}: {e}")
			# Add empty predictions for failed batch
			predictions.extend([""] * len(batch_audios))
	
	# Calculate metrics
	valid_indices = [i for i, p in enumerate(predictions) if p != ""]
	valid_predictions = [predictions[i] for i in valid_indices]
	valid_sentences = [sentences[i] for i in valid_indices]
	
	if len(valid_predictions) == 0:
		print("ERROR: No valid predictions!")
		return None, None
	
	wer_metric = evaluate.load("wer")
	cer_metric = evaluate.load("cer")
	wer = wer_metric.compute(predictions=valid_predictions, references=valid_sentences)
	cer = cer_metric.compute(predictions=valid_predictions, references=valid_sentences)
	
	# Print results
	print("\nEVALUATION RESULTS:")
	print(f"  Valid predictions: {len(valid_predictions)}/{len(sentences)}")
	print(f"  WER: {wer:.4f}")
	print(f"  CER: {cer:.4f}")
	
	# Show examples
	print("\nSAMPLE PREDICTIONS (first 5):")
	for i in range(min(5, len(sentences))):
		print(f"\nSample {i+1}:")
		print(f"  Ground truth: '{sentences[i]}'")
		print(f"  Prediction:   '{predictions[i]}'")
	
	# Save results
	predictions_file = os.path.join(output_dir, "predictions.txt")
	with open(predictions_file, 'w', encoding='utf-8') as f:
		for i in range(len(sentences)):
			f.write(f"Ground truth: {sentences[i]}\n")
			f.write(f"Prediction:   {predictions[i]}\n\n")
	
	results_file = os.path.join(output_dir, "evaluation_results.txt")
	with open(results_file, 'w', encoding='utf-8') as f:
		f.write(f"Model: {pretrained_model_name}\n")
		f.write(f"Valid predictions: {len(valid_predictions)}/{len(sentences)}\n")
		f.write(f"WER: {wer:.4f}\n")
		f.write(f"CER: {cer:.4f}\n")
	
	print(f"\nResults saved to: {predictions_file} and {results_file}")
	
	return wer, cer
	
def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type=str, default="asr_model/")
	parser.add_argument("--test_file", type=str, default="sentence_test_1.csv", help="Test data CSV file")
	parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-960h", 
	                    help="Pretrained model from HuggingFace (e.g., facebook/wav2vec2-large-960h, facebook/wav2vec2-base-960h)")
	parser.add_argument("--output_dir", type=str, default="asr_model/pretrained_eval/", 
	                    help="Directory to save evaluation results")
	args = parser.parse_args()

	data_path = args.data_path
	test_file = args.test_file
	pretrained_model = "facebook/" + args.pretrained_model
	output_dir = args.output_dir

	# Create output directory with model name
	model_name_clean = pretrained_model
	output_dir_full = os.path.join(output_dir, model_name_clean)
	
	print('Loading test data...')
	test_data = get_data_simple(data_path, test_file)

	print(f"Test samples: {len(test_data)}")
	if len(test_data) > 0:
		print(f"Sample audio length: {len(test_data[0]['audio']['array'])}")
		print(f"Sample transcript: {test_data[0]['sentence']}")

	# Evaluate pretrained model
	wer, cer = evaluate_pretrained_model(pretrained_model, test_data, output_dir_full)
	
	print(f"\nFINAL SUMMARY:")
	print(f"Model: {pretrained_model}")
	print(f"WER: {wer:.4f}")
	print(f"CER: {cer:.4f}")

if __name__ == "__main__":
	main()
	print("Script finished running")
