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
from transformers import AutoModelForCTC  
from pyctcdecode import build_ctcdecoder 

if not os.path.exists('asr_model/'):
	os.system('mkdir asr_model/')

def read_audio(fname):
	""" Load an audio file and return PCM along with the sample rate """
	wav, sr = sf.read(fname)
	return wav, sr

chars_to_ignore_regex = '[\,\?\!\-\;\"\(\)\&\-\>\[\]\_]'

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

@dataclass
class DataCollatorCTCWithPadding:
	"""
	Data collator that will dynamically pad the inputs received.
	"""
	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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

def get_data_reg(data_path, file, pretrained_model, seed, cmu_audio=False, childes_lm=False, cmu_lm=False):
	## Define n-gram LM file
	lm_text = data_path + 'random/' + '/' + seed + '/lm_text.txt'
	lm = data_path + 'random/' + '/' + seed + '/lm.arpa'
	lm_vocab = data_path + 'random/' + '/' + seed + '/lm_vocab.txt'

	data_path_file = data_path + file
	data_path_file = data_path_file.replace(u'\xa0', u'')
	original_data = pd.read_csv(data_path_file)
	path_list = original_data['path'].tolist()
	transcript_list = original_data['transcript'].tolist()

	if cmu_audio:
		## Add CMU Kids data for ASR training
		cmu_path = '/blue/liu.ying/Storiza_speech_modeling/asr_model/cmu_kids.csv'
		cmu_data = pd.read_csv(cmu_path)
		cmu_audio_list = cmu_data['path'].tolist()
		cmu_transcript_list = cmu_data['transcript'].tolist()
		path_list = path_list + cmu_audio_list
		transcript_list = transcript_list + cmu_transcript_list
		print(f"Added CMU Kids data: {len(cmu_audio_list)} samples")
		shuffle_indices = list(range(len(path_list)))
		shuffle(shuffle_indices)
		shuffle_indices = sorted(shuffle_indices, key=lambda k: path_list[k])

	## Getting audio data
	data = []
	duration = 0
	words = []
	cleaned_transcripts = []

	for i in range(len(path_list)):
		wav_path = path_list[i]
		try:
			transcript = clean_sent(transcript_list[i])
			cleaned_transcripts.append(transcript)
			signal, samplerate = librosa.load(wav_path, sr=16000)
			
			if len(signal.shape) > 1 and signal.shape[1] == 2:
				signal = np.average(signal, axis=1)

			entry = {}
			duration += len(signal) / samplerate
			words = words + transcript.split()
			entry["sentence"] = transcript.replace("\n", " ")				
			entry["audio"] = {"sampling_rate" : samplerate, "array" : signal}
			data.append(entry)
		except Exception as e:
			print(f"Error processing {wav_path}: {e}")
			continue

	print(f"Total duration: {duration:.2f} seconds")

	if 'train' in file:
		## Getting text data for training n-gram LM
		print('\nBUILDING LANGUAGE MODEL')
		print(f"LM files: {lm}, {lm_text}")
	
		with open(lm_text, 'w', encoding="utf-8") as f:
			for tok in cleaned_transcripts:
				f.write(tok + '\n')
		
		if childes_lm:
			print("Adding CHILDES data to LM training")
			childes_path = '/blue/liu.ying/Storiza_speech_modeling/asr_model/CHILDES_child_transcripts.txt'
			with open(childes_path, 'r', encoding="utf-8") as f:
				lines = f.readlines()
			with open(lm_text, 'a', encoding="utf-8") as f:
				for line in lines:
					cleaned_line = clean_sent(line)
					f.write(cleaned_line + '\n')
		
		if cmu_lm:
			print("Adding CMU Kids data to LM training")
			cmu_path = '/blue/liu.ying/Storiza_speech_modeling/asr_model/cmu_kids.csv'
			cmu_data = pd.read_csv(cmu_path)
			cmu_transcripts = cmu_data['transcript'].tolist()
			with open(lm_text, 'a', encoding="utf-8") as f:
				for line in cmu_transcripts:
					cleaned_line = clean_sent(line)
					f.write(cleaned_line + '\n')

		## Training an n-gram LM with SRILM
		os.system('module load gcc')
		os.system('module load perl/5.20.0')
		os.system(f'/blue/liu.ying/asr_resource/kaldi/tools/srilm/bin/i686-m64/ngram-count -order 3 -unk -write-vocab {lm_vocab} -wbdiscount -text {lm_text} -lm {lm}')

	return data, lm

def train(data_path, train_data, test_data, pretrained_model, seed):
	print("Creating vocabulary...")
	vocab_train = set(y for x in train_data for y in x["sentence"])
	vocab_test = set(y for x in test_data for y in x["sentence"])
	vocab = vocab_train.union(vocab_test)
	
	if "\n" in vocab:
		vocab.remove("\n")
	
	vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	vocab_dict["[UNK]"] = len(vocab_dict)
	vocab_dict["[PAD]"] = len(vocab_dict)
	
	print(f'Vocabulary size: {len(vocab_dict)}')
	for v, k in vocab_dict.items():
		print(v, k)

	with open(data_path + 'random/' + seed + '/' + pretrained_model + '/vocab.json', 'w', encoding="utf-8") as vocab_file:
		json.dump(vocab_dict, vocab_file, ensure_ascii=False)
	
	## Creation of the tokeniser
	print("Setting up tokenizer...")
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(data_path + 'random/' + seed + '/' + pretrained_model + '/', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	
	repo_name = f'asr_model/random/{seed}/{pretrained_model}/'
	print(f"Model will be saved to: {repo_name}")
	
	if not os.path.exists(repo_name):
		os.makedirs(repo_name)
		
	tokenizer.save_pretrained(repo_name)

	## Extraction of speech features
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	# Debug vocabulary alignment
	print(f"Processor vocab size: {len(processor.tokenizer)}")
	sample_text = train_data[0]["sentence"]
	print(f"Sample text: '{sample_text}'")
	tokenized = processor.tokenizer(sample_text)
	print(f"Tokenized IDs: {tokenized.input_ids}")
	print(f"Decoded back: '{processor.tokenizer.decode(tokenized.input_ids)}'")

	def compute_metrics(pred):
		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)
		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
		pred_str = processor.batch_decode(pred_ids)
		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
		wer = wer_metric.compute(predictions=pred_str, references=label_str)
		return {"wer": wer}

	def prepare_dataset(batch):
		audio = batch["audio"]
		batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
		batch["input_length"] = len(batch["input_values"])
		
		with processor.as_target_processor():
			batch["labels"] = processor(batch["sentence"]).input_ids
		return batch

	## Setting up data for training
	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	wer_metric = evaluate.load("wer")
	
	train_data_temp = list(map(prepare_dataset, train_data))
	train_data_processed = [tok for tok in train_data_temp if tok is not None]
	test_data_temp = list(map(prepare_dataset, test_data))
	test_data_processed = [tok for tok in test_data_temp if tok is not None]

	print(f"Prepared train samples: {len(train_data_processed)}")
	print(f"Prepared test samples: {len(test_data_processed)}")
 
	print("Preparing model...")
	## Training
	model = Wav2Vec2ForCTC.from_pretrained(
		"facebook/" + pretrained_model, 
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
	
	from transformers import TrainingArguments

	epochs = 30
	batch_size = 8

	training_args = TrainingArguments(
		output_dir=repo_name,
		group_by_length=True,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=8,
		gradient_accumulation_steps=2,
		eval_strategy="steps",
		num_train_epochs=epochs,
		gradient_checkpointing=True,
		fp16=True,
		save_steps=500,
		eval_steps=500,
		logging_steps=500,
		load_best_model_at_end=True,
		warmup_steps=500,
		learning_rate=1e-4,
		metric_for_best_model="wer",
		save_total_limit=2,
		greater_is_better=False,
		push_to_hub=False,
	)

	from transformers import Trainer
	print("Starting training...")

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_data_processed,
		eval_dataset=test_data_processed,
		tokenizer=processor.feature_extractor
	)

	trainer.train()
	print("Training completed!")

	return train_data, test_data, repo_name, vocab_dict, processor

def evaluate_with_pyctcdecode(model_path, test_data, vocab_dict, lm_path=None):
	"""
	Evaluate the trained model using pyctcdecode
	"""
	print("\n" + "="*50)
	print("EVALUATING WITH PYCTCDECODE")
	print("="*50)
	
	# Load the best checkpoint
	checkpoints = [x for x in os.listdir(model_path) if "checkpoint" in x]
	if checkpoints:
		checkpoint = max(checkpoints, key=lambda y: int(y.split('-')[1]))
		checkpoint_path = os.path.join(model_path, checkpoint)
		print(f"Loading checkpoint: {checkpoint_path}")
	else:
		checkpoint_path = model_path
		print(f"Loading model from: {model_path}")
	
	# Copy tokenizer files if needed
	required_files = ["tokenizer_config.json", "vocab.json"]
	for file in required_files:
		if not os.path.exists(os.path.join(checkpoint_path, file)):
			src = os.path.join(model_path, file)
			dst = os.path.join(checkpoint_path, file)
			if os.path.exists(src):
				shutil.copy(src, dst)
				print(f"Copied {file} to checkpoint directory")

	# Load model and processor
	model = AutoModelForCTC.from_pretrained(checkpoint_path).to("cuda")
	processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
	
	# Build decoder - use processor vocabulary to match model
	vocab = processor.tokenizer.get_vocab()
	# Sort by token ID to match model output order
	vocab_list = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
	
	# Check if LM exists and use it
	use_lm = lm_path and os.path.exists(lm_path)
	if use_lm:
		print(f"Using language model: {lm_path}")
		decoder = build_ctcdecoder(
			vocab_list,
			kenlm_model_path=lm_path,
			alpha=0.5,
			beta=1.5
		)
	else:
		print("Using decoder without language model")
		decoder = build_ctcdecoder(
			vocab_list,
			kenlm_model_path=None,
			alpha=0.5,
			beta=1.5
		)
	
	# Prepare data for evaluation
	signals = [x["audio"]["array"] for x in test_data]
	sentences = [x["sentence"] for x in test_data]
	
	print(f"Evaluating {len(signals)} samples...")
	
	# Predictions using processor.decode (same as training)
	processor_preds = []
	# Predictions using pyctcdecode
	pyctc_preds = []
	
	model.eval()
	for i in tqdm(range(len(signals))):
		signal = signals[i:i+1]
		
		# Handle stereo audio
		if len(signal[0].shape) > 1 and signal[0].shape[1] == 2:
			signal = [np.average(signal[0], axis=1)]
		
		# Get model predictions
		inputs = processor(signal, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
		
		with torch.no_grad():
			logits = model(inputs).logits
		
		# Processor decode (same as training)
		pred_ids = torch.argmax(logits, dim=-1).cpu()
		processor_pred = processor.batch_decode(pred_ids)[0]
		processor_preds.append(processor_pred)
		
		# pyctcdecode
		logits_np = logits.cpu().numpy()[0]
		pyctc_pred = decoder.decode(logits_np)
		pyctc_preds.append(pyctc_pred)
	
	# Calculate metrics
	wer_metric = evaluate.load("wer")
	cer_metric = evaluate.load("cer")
	
	# Processor results (should match training eval)
	processor_wer = wer_metric.compute(predictions=processor_preds, references=sentences)
	processor_cer = cer_metric.compute(predictions=processor_preds, references=sentences)
	
	# pyctcdecode results
	pyctc_wer = wer_metric.compute(predictions=pyctc_preds, references=sentences)
	pyctc_cer = cer_metric.compute(predictions=pyctc_preds, references=sentences)
	
	# Print results
	print("\nEVALUATION RESULTS:")
	print(f"Processor decode (same as training):")
	print(f"  WER: {processor_wer:.4f}")
	print(f"  CER: {processor_cer:.4f}")
	print(f"\nPyctcdecode ({'with LM' if use_lm else 'no LM'}):")
	print(f"  WER: {pyctc_wer:.4f}")
	print(f"  CER: {pyctc_cer:.4f}")
	
	# Show some examples
	print("\nSAMPLE PREDICTIONS (first 3):")
	for i in range(min(3, len(sentences))):
		print(f"\nSample {i+1}:")
		print(f"  Ground truth: '{sentences[i]}'")
		print(f"  Processor:    '{processor_preds[i]}'")
		print(f"  Pyctcdecode:  '{pyctc_preds[i]}'")
	
	# Generate predictions
	predictions_file = os.path.join(model_path, "predictions.txt")
	with open(predictions_file, 'w') as f:
		for i in range(len(sentences)):
			f.write(f"Ground truth: {sentences[i]}\n")
		#	f.write(f"Processor:    {processor_preds[i]}\n")
			f.write(f"Pyctcdecode:  {pyctc_preds[i]}\n\n")

	# Save results
	results_file = os.path.join(model_path, "evaluation_results.txt")
	with open(results_file, 'w') as f:
		f.write(f"Processor decode WER: {processor_wer:.4f}\n")
		f.write(f"Processor decode CER: {processor_cer:.4f}\n")
		f.write(f"Pyctcdecode WER: {pyctc_wer:.4f}\n")
		f.write(f"Pyctcdecode CER: {pyctc_cer:.4f}\n")
		f.write(f"Language model used: {use_lm}\n")
	
	return processor_wer, pyctc_wer

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type=str, default="asr_model/")
	parser.add_argument("--seed", type=str, default="1")
	parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-xlsr-53")
	parser.add_argument("--skip_training", action="store_true", help="Skip training and only run evaluation")
	parser.add_argument("--cmu_audio", action="store_true", help="Include CMU Kids audio data for training")
	parser.add_argument("--childes_lm", action="store_true", help="Include CHILDES data for LM training")
	parser.add_argument("--cmu_lm", action="store_true", help="Include CMU Kids data for LM training")
	args = parser.parse_args()

	data_path = args.data_path
	seed = args.seed
	pretrained_model = args.pretrained_model

	# Create directories
	try:
		if not args.skip_training:
			os.system(f'rm -r {data_path}random/{seed}/*')
			print('Removed old models')
	except:
		pass

	for dir_path in [f'{data_path}random/', f'{data_path}random/{seed}']:
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	print('Loading data...')
	train_data, lm_path = get_data_reg(data_path, f'sentence_train_{seed}.csv', pretrained_model, seed, args.cmu_audio, args.childes_lm, args.cmu_lm)
	test_data, _ = get_data_reg(data_path, f'sentence_test_{seed}.csv', pretrained_model, seed)

	print(f"Train samples: {len(train_data)}")
	print(f"Test samples: {len(test_data)}")
	print(f"Sample audio length: {len(train_data[0]['audio']['array'])}")
	print(f"Sample transcript: {train_data[0]['sentence']}")

	model_path = f'asr_model/random/{seed}/{pretrained_model}'

	if not args.skip_training:
		# Train the model
		train_data, test_data, repo_name, vocab_dict, processor = train(data_path, train_data, test_data, pretrained_model, seed)
		model_path = repo_name
	else:
		# Load existing vocab for evaluation only
		with open('vocab.json', 'r') as f:
			vocab_dict = json.load(f)
		print("Skipping training, loading existing model...")

	# Evaluate with pyctcdecode
	processor_wer, pyctc_wer = evaluate_with_pyctcdecode(model_path, test_data, vocab_dict, lm_path)
	
	print(f"\nFINAL SUMMARY:")
	print(f"Training eval WER (processor): {processor_wer:.4f}")
	print(f"Pyctcdecode WER: {pyctc_wer:.4f}")

if __name__ == "__main__":
	main()
	print("Script finished running")

