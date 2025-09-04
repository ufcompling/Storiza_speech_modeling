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
from datasets import Dataset

if not os.path.exists('asr_model/'):
	os.system('mkdir asr_model/')

def read_audio(fname):
	""" Load an audio file and return PCM along with the sample rate """
	wav, sr = sf.read(fname)
	return wav, sr

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\"\%\'\"\�\]\[]'

def clean_sent(sent):
	sent = re.sub(chars_to_remove_regex, '', sent).lower()
	return sent

@dataclass
class DataCollatorCTCWithPadding:
	"""
	Data collator that will dynamically pad the inputs received.
	"""
	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# Filter out None features and ensure required keys exist
		valid_features = []
		for f in features:
			if f is not None and 'input_values' in f and 'labels' in f:
				valid_features.append(f)
		
		if len(valid_features) == 0:
			print("Warning: No valid features in batch")
			return None
			
		features = valid_features
		
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				return_tensors="pt",
			)

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		batch["labels"] = labels
		return batch

def get_data_reg(data_path, file, pretrained_model, seed):
	lm_text = data_path + 'random/' + '/' + seed + '/' + pretrained_model + '_lm.txt'
	lm = data_path + 'random/' + '/' + seed + '/' + pretrained_model + '_lm.arpa'
	lm_vocab = data_path + 'random/' + '/' + seed + '/' + pretrained_model + '_vocab.txt'

	data_path = data_path + file
	data_path = data_path.replace(u'\xa0', u'')
	
	print(f"Looking for data file: {data_path}")
	
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Data file not found: {data_path}")
	
	original_data = pd.read_csv(data_path)
	path_list = original_data['path'].tolist()
	transcript_list = original_data['transcript'].tolist()

	if 'train' in file:
		print('BUILDING LANGUAGE MODEL')
		print('\n')
		## Define n-gram LM file	
		print(lm, lm_text)
		print('\n')
		with open(lm_text, 'w', encoding="utf-8") as f:
			for tok in transcript_list:
				f.write(tok + '\n')

	## Training an n-gram LM with SRILM
	os.system('module load gcc')
	os.system('module load perl/5.20.0')
	os.system('/blue/liu.ying/asr_resource/kaldi/tools/srilm/bin/i686-m64/ngram-count -order 3 -unk -write-vocab ' + lm_vocab + ' -wbdiscount -text ' + lm_text + ' -lm ' + lm)


	## Getting audio data
	data = []
	duration = 0
	words = []
	failed_files = 0

	print(f"Processing {len(path_list)} audio files...")

	for i in range(len(path_list)):
		wav_path = path_list[i]
		transcript = transcript_list[i]
		
		# Skip if transcript is NaN or empty
		if pd.isna(transcript) or str(transcript).strip() == "":
			print(f"Skipping {wav_path}: empty transcript")
			failed_files += 1
			continue
		
		transcript = clean_sent(str(transcript))
		
		# Skip if cleaned transcript is empty
		if not transcript or transcript.strip() == "":
			print(f"Skipping {wav_path}: empty after cleaning")
			failed_files += 1
			continue
		
		try:
			# Check if file exists
			if not os.path.exists(wav_path):
				print(f"Skipping {wav_path}: file not found")
				failed_files += 1
				continue
				
			signal, samplerate = librosa.load(wav_path, sr=16000)
			
			# Handle different audio formats
			if len(signal) == 0:
				print(f"Skipping {wav_path}: empty audio")
				failed_files += 1
				continue
			
			# Handle stereo to mono conversion
			if signal.ndim > 1:
				if signal.shape[0] == 2:  # Stereo
					signal = np.average(signal, axis=0)
				else:
					signal = signal.flatten()
			
			# Check minimum duration (e.g., at least 0.1 seconds)
			if len(signal) < 1600:  # 0.1 seconds at 16kHz
				print(f"Skipping {wav_path}: too short ({len(signal)/16000:.3f}s)")
				failed_files += 1
				continue
			
			entry = {}
			duration += len(signal) / samplerate
			words = words + transcript.split()
			entry["sentence"] = transcript.replace("\n", " ")				
			entry["audio"] = {"sampling_rate" : samplerate, "array" : signal}
			data.append(entry)
			
		except Exception as e:
			print(f"Error processing {wav_path}: {e}")
			failed_files += 1
			continue

	print(f"Successfully processed: {len(data)} files")
	print(f"Failed/Skipped: {failed_files} files")
	print(f"Total duration: {duration/3600:.2f} hours")
	
	if len(data) == 0:
		raise ValueError("No valid audio files found after processing!")
		
	return data

def prepare_dataset(batch, processor):
	"""Prepare dataset with error handling"""
	try:
		# Check if batch has required fields
		if not batch or "audio" not in batch or "sentence" not in batch:
			print("Batch missing required fields")
			return None
			
		audio = batch["audio"]
		if not audio or "array" not in audio or "sampling_rate" not in audio:
			print("Audio data missing required fields")
			return None
		
		# Check audio array
		audio_array = audio["array"]
		if audio_array is None or len(audio_array) == 0:
			print("Empty audio array")
			return None
		
		# Process audio
		processed_audio = processor(audio_array, sampling_rate=audio["sampling_rate"])
		if not processed_audio or "input_values" not in processed_audio:
			print("Failed to process audio")
			return None
			
		batch["input_values"] = processed_audio.input_values[0]
		batch["input_length"] = len(batch["input_values"])
		
		# Process text
		with processor.as_target_processor():
			processed_text = processor(batch["sentence"])
			if not processed_text or "input_ids" not in processed_text:
				print("Failed to process text")
				return None
			batch["labels"] = processed_text.input_ids
		
		return batch
	except Exception as e:
		print(f"Error processing batch: {e}")
		return None

def safe_load_checkpoint(model, checkpoint_path):
	"""
	Safely load checkpoint, handling vocab size mismatches
	"""
	if not os.path.exists(checkpoint_path):
		print(f"No checkpoint found at {checkpoint_path}")
		return False
	
	try:
		print(f"Attempting to load checkpoint from {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		
		# Get current model state
		model_state = model.state_dict()
		
		# Filter checkpoint to match current model
		filtered_checkpoint = {}
		mismatched_keys = []
		
		for key, value in checkpoint.items():
			if key in model_state:
				if model_state[key].shape == value.shape:
					filtered_checkpoint[key] = value
				else:
					mismatched_keys.append(key)
					print(f"Size mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
			else:
				print(f"Key {key} not found in current model")
		
		if mismatched_keys:
			print(f"Found {len(mismatched_keys)} mismatched parameters - these will be randomly initialized")
			if any('lm_head' in key for key in mismatched_keys):
				print("Language model head size mismatch - expected when vocabulary changes")
		
		# Load compatible parameters
		model.load_state_dict(filtered_checkpoint, strict=False)
		print(f"Successfully loaded {len(filtered_checkpoint)} compatible parameters")
		return True
		
	except Exception as e:
		print(f"Error loading checkpoint: {e}")
		return False

def train(data_path, train_data, test_data, pretrained_model, seed, resume_checkpoint=None, ignore_vocab_mismatch=False):

	print("creating vocab")
	vocab_train = set(y for x in train_data for y in x["sentence"])
	vocab_test = set(y for x in test_data for y in x["sentence"])
	vocab = vocab_train.union(vocab_test)
	if "\n" in vocab:
		vocab.remove("\n")
	vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]

	print(f"Vocabulary size: {len(vocab_dict)}")

	vocab_file = data_path + '/random/' + seed + '/' + 'vocab.json'
	with open(vocab_file, 'w', encoding="utf-8") as vocab_file:
		json.dump(vocab_dict, vocab_file, ensure_ascii=False)
	
	print("setting up tokeniser")
#	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(data_path + '/random/' + seed + '/', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
#	tokenizer.save_pretrained(data_path + '/random/' + seed + "/tokenizer_dir")
#	tokenizer_path = data_path + '/random/' + seed + "/tokenizer_dir"
	# Then you can load it with from_pretrained
#	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_path)
	print("tokeniser saved")
	repo_name = data_path + '/random/' + seed + '/' + pretrained_model
	print(repo_name) 
	tokenizer.save_pretrained(repo_name)

	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	def compute_metrics(pred):
		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)
		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
		pred_str = processor.batch_decode(pred_ids)
		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
		wer = wer_metric.compute(predictions=pred_str, references=label_str)
		return {"wer": wer}

	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	wer_metric = evaluate.load("wer")
	
	# Process the data and filter out None values
	print("Processing training data...")
	train_data_processed = []
	train_failed = 0
	
	for i, item in enumerate(train_data):
		processed_item = prepare_dataset(item, processor)
		if processed_item is not None:
			train_data_processed.append(processed_item)
		else:
			train_failed += 1
			print(f"Failed to process training item {i}")
	
	print(f"Training data: {len(train_data_processed)} successful, {train_failed} failed")
	
	if len(train_data_processed) == 0:
		raise ValueError("No valid training samples after processing!")

	print("Processing test data...")
	test_data_processed = []
	test_failed = 0
	
	for i, item in enumerate(test_data):
		processed_item = prepare_dataset(item, processor)
		if processed_item is not None:
			test_data_processed.append(processed_item)
		else:
			test_failed += 1
			print(f"Failed to process test item {i}")
	
	print(f"Test data: {len(test_data_processed)} successful, {test_failed} failed")
	
	# Convert to HuggingFace Dataset objects
	print("Converting to Dataset objects...")
	train_dataset = Dataset.from_list(train_data_processed)
	
	# Only create test dataset if we have valid test data
	test_dataset = Dataset.from_list(test_data_processed) if test_data_processed else None
	
	print(f"Training samples: {len(train_dataset)}")
	print(f"Test samples: {len(test_dataset) if test_dataset else 0}")
	
	# Verify dataset structure
	print("Checking dataset structure...")
	sample = train_dataset[0]
	print(f"Sample keys: {list(sample.keys())}")
	if 'input_values' in sample:
		print(f"Input values shape: {np.array(sample['input_values']).shape}")
	if 'labels' in sample:
		print(f"Labels shape: {np.array(sample['labels']).shape}")
 
	print("preparing model")
	model = Wav2Vec2ForCTC.from_pretrained(
		"facebook/" + pretrained_model, 
		force_download=True,
		cache_dir=None,
		attention_dropout=0.0,
		hidden_dropout=0.0,
		feat_proj_dropout=0.0,
		mask_time_prob=0.05,
		layerdrop=0.0,
		ctc_loss_reduction="mean", 
		pad_token_id=processor.tokenizer.pad_token_id,
		vocab_size=len(processor.tokenizer),
	)
	model.config.ctc_zero_infinity = True
	model.freeze_feature_extractor()
	
	from transformers import TrainingArguments, Trainer

	epochs = 30
	batch_size = 16

	training_args = TrainingArguments(
		output_dir=repo_name,
		group_by_length=True,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=8,
		gradient_accumulation_steps=2,
		eval_strategy="steps" if test_dataset else "no",
		num_train_epochs=epochs,
		gradient_checkpointing=True,
		fp16=torch.cuda.is_available(),
		save_steps=50,
		eval_steps=50 if test_dataset else None,
		logging_steps=50,
		learning_rate=3e-4,
		metric_for_best_model="wer" if test_dataset else None,
		save_total_limit=2,
		greater_is_better=False,
		push_to_hub=False,
		dataloader_drop_last=False,  # Don't drop incomplete batches
		remove_unused_columns=False,  # Keep all columns
	)

	if torch.cuda.is_available():
		device = torch.device("cuda")
		model = model.to(device)
		print(f"Model moved to: {device}")
	else:
		print("CUDA not available - using CPU")

	# Handle checkpoint loading/resuming
	checkpoint_to_load = None
	if resume_checkpoint:
		checkpoint_to_load = resume_checkpoint
	else:
		# Check for automatic checkpoint in output directory
		auto_checkpoint = os.path.join(repo_name, 'pytorch_model.bin')
		if os.path.exists(auto_checkpoint):
			checkpoint_to_load = auto_checkpoint
	
	if checkpoint_to_load:
		print(f"Found checkpoint: {checkpoint_to_load}")
		if ignore_vocab_mismatch:
			print("Attempting to load checkpoint with vocab mismatch handling...")
			checkpoint_loaded = safe_load_checkpoint(model, checkpoint_to_load)
			if checkpoint_loaded:
				print("Checkpoint loaded successfully with vocab mismatch handling")
			else:
				print("Checkpoint loading failed - starting fresh training")
		else:
			# Try normal loading first
			try:
				state_dict = torch.load(checkpoint_to_load, map_location='cpu')
				model.load_state_dict(state_dict)
				print("Checkpoint loaded successfully")
			except RuntimeError as e:
				if "size mismatch" in str(e):
					print(f"Vocabulary size mismatch detected: {e}")
					print("Options:")
					print("1. Use --ignore_vocab_mismatch flag to load compatible layers only")
					print("2. Delete the existing checkpoint to start fresh")
					print("3. Use a dataset with the same vocabulary size")
					raise RuntimeError(f"Vocabulary mismatch. {str(e)}\nUse --ignore_vocab_mismatch to continue.")
				else:
					raise
	else:
		print("No checkpoint found - starting fresh training")

	def debug_checkpoint_sources(repo_name):
		"""Debug function to find where checkpoints might be coming from"""
		print("\n=== CHECKPOINT DEBUG ===")

		# Check the main output directory
		if os.path.exists(repo_name):
			print(f"Output directory exists: {repo_name}")
			for item in os.listdir(repo_name):
				item_path = os.path.join(repo_name, item)
				if os.path.isdir(item_path):
					print(f"  Directory: {item}")
					if item.startswith('checkpoint-'):
						print(f"    FOUND CHECKPOINT DIR: {item}")
						# List contents
						for subitem in os.listdir(item_path):
							print(f"      - {subitem}")
				else:
					print(f"  File: {item}")
					if item in ['pytorch_model.bin', 'config.json', 'trainer_state.json']:
						print(f"    FOUND CHECKPOINT FILE: {item}")
		else:
			print(f"Output directory does not exist: {repo_name}")
		
		# Check current directory for any checkpoint files
		print(f"\nCurrent directory: {os.getcwd()}")
		current_files = [f for f in os.listdir('.') if 'checkpoint' in f.lower() or f.endswith('.bin')]
		if current_files:
			print("Checkpoint-like files in current directory:")
			for f in current_files:
				print(f"  - {f}")
		
		# Check for Hugging Face cache
		cache_dirs = [
			os.path.expanduser("~/.cache/huggingface/transformers/"),
			os.path.expanduser("~/.cache/huggingface/hub/"),
		]
		for cache_dir in cache_dirs:
			if os.path.exists(cache_dir):
				print(f"\nHugging Face cache directory: {cache_dir}")
				try:
					cache_items = os.listdir(cache_dir)[:10]  # First 10 items
					print(f"  Contains {len(os.listdir(cache_dir))} items (showing first 10):")
					for item in cache_items:
						print(f"    - {item}")
				except:
					print("  (Cannot list contents)")
		
		print("=== END CHECKPOINT DEBUG ===\n")

	debug_checkpoint_sources(repo_name)

	print("Starting training...")
	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics if test_dataset else None,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		tokenizer=processor.feature_extractor
	)

	try:
		trainer.train()
		print("Training completed successfully!")
		
		# Save the final model
		trainer.save_model()
		processor.save_pretrained(repo_name)
		print(f"Model saved to {repo_name}")
		
	except Exception as e:
		print(f"Training error: {e}")
		# Debug: print dataset info
		print("Debugging dataset...")
		for i in range(min(3, len(train_dataset))):
			sample = train_dataset[i]
			print(f"Sample {i}: keys = {list(sample.keys())}")
			for key, value in sample.items():
				if isinstance(value, (list, np.ndarray)):
					print(f"  {key}: shape = {np.array(value).shape}, type = {type(value)}")
				else:
					print(f"  {key}: value = {value}, type = {type(value)}")
		raise

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type=str, default="asr_model/")
	parser.add_argument("--seed", type=str, default="1")
	parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-960h")
	parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
						help="Path to checkpoint to resume from")
	parser.add_argument("--ignore_vocab_mismatch", action="store_true",
						help="Continue training even with vocabulary size mismatch")

	args = parser.parse_args()

	# Removing previous models

	data_path = args.data_path + '/'
	seed = args.seed
	pretrained_model = args.pretrained_model

	os.system('rm -r ' + data_path + 'random/' + seed + '/*')

	if not os.path.exists(data_path + 'random/'):
		os.system('mkdir ' + data_path + 'random/')

	if not os.path.exists(data_path + 'random/' + seed):
		os.system('mkdir ' + data_path + 'random/' + seed)

	print('loading data')
	try:
		train_data = get_data_reg(data_path, 'sentence_train_' + seed + '.csv', pretrained_model, seed)
		test_data = get_data_reg(data_path, 'sentence_test_' + seed + '.csv', pretrained_model, seed)
		
		print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
		
		# Pass resume arguments to train function
		train(data_path, train_data, test_data, pretrained_model, seed, 
			  resume_checkpoint=args.resume_from_checkpoint,
			  ignore_vocab_mismatch=args.ignore_vocab_mismatch)
		
	except Exception as e:
		print(f"Error: {e}")
		print("Please check:")
		print(f"1. File exists: {data_path}sentence_train_{seed}.csv")
		print(f"2. File exists: {data_path}sentence_test_{seed}.csv")
		print("3. CSV files have 'path' and 'transcript' columns")
		print("4. Audio files referenced in the CSV exist")
		print("\nIf you're getting vocabulary size mismatch errors:")
		print("- Use --ignore_vocab_mismatch to continue with different vocab size")
		print("- Or delete the existing checkpoint to start fresh")
		raise

if __name__ == "__main__":
	main()
	print("train_random_w2v.py finished running")