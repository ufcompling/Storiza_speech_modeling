import numpy as np
import re, os
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor, HubertForCTC
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
import random
import scipy.signal

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

# ============= DATA AUGMENTATION METHODS =============

class AudioAugmentation:
    """Audio augmentation methods for ASR training"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def speed_perturbation(self, audio, min_speed=0.9, max_speed=1.1):
        """Change playback speed without changing pitch"""
        speed_factor = random.uniform(min_speed, max_speed)
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def pitch_shift(self, audio, min_steps=-2, max_steps=2):
        """Shift pitch without changing speed"""
        steps = random.uniform(min_steps, max_steps)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio + noise
    
    def time_masking(self, audio, max_mask_pct=0.1, n_masks=1):
        """Mask random time segments"""
        audio_copy = audio.copy()
        for _ in range(n_masks):
            mask_length = int(len(audio) * random.uniform(0.01, max_mask_pct))
            mask_start = random.randint(0, max(1, len(audio) - mask_length))
            audio_copy[mask_start:mask_start + mask_length] = 0
        return audio_copy
    
    def frequency_masking(self, audio, max_freq_mask=0.15):
        """Apply frequency masking in spectral domain"""
        # Convert to spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Mask random frequency bands
        freq_bins = magnitude.shape[0]
        mask_size = int(freq_bins * random.uniform(0.05, max_freq_mask))
        mask_start = random.randint(0, max(1, freq_bins - mask_size))
        
        magnitude[mask_start:mask_start + mask_size, :] *= 0.1
        
        # Convert back to audio
        masked_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(masked_stft)
    
    def dynamic_range_compression(self, audio, ratio=4.0, threshold=0.1):
        """Apply dynamic range compression"""
        audio_copy = audio.copy()
        mask = np.abs(audio_copy) > threshold
        audio_copy[mask] = threshold + (audio_copy[mask] - threshold) / ratio
        return audio_copy
    
    def reverb_simulation(self, audio, room_size=0.5, damping=0.5):
        """Simple reverb simulation using convolution"""
        # Create simple impulse response
        ir_length = int(0.3 * self.sr)  # 300ms reverb
        ir = np.random.exponential(scale=room_size, size=ir_length)
        ir *= np.exp(-np.linspace(0, damping * 10, ir_length))
        ir /= np.sum(ir)  # normalize
        
        # Convolve with impulse response
        reverb_audio = np.convolve(audio, ir, mode='same')
        
        # Mix with original (50% wet, 50% dry)
        return 0.5 * audio + 0.5 * reverb_audio
    
    def band_pass_filter(self, audio, low_freq=300, high_freq=7000):
        """Apply band-pass filter to simulate telephone/radio quality"""
        nyquist = self.sr // 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth band-pass filter
        sos = scipy.signal.butter(5, [low, high], btype='band', output='sos')
        return scipy.signal.sosfilt(sos, audio)
    
    def volume_perturbation(self, audio, min_gain=0.7, max_gain=1.3):
        """Random volume changes"""
        gain = random.uniform(min_gain, max_gain)
        return audio * gain
    
    def apply_augmentation(self, audio, augmentation_config=None):
        """Apply random augmentations based on config"""
        if augmentation_config is None:
            augmentation_config = {
                'speed_perturbation': 0.5,
                'pitch_shift': 0.3,
                'add_noise': 0.4,
                'time_masking': 0.3,
                'frequency_masking': 0.2,
                'dynamic_range_compression': 0.2,
                'reverb_simulation': 0.2,
                'band_pass_filter': 0.1
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				return_tensors="pt",
			)

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		batch["labels"] = labels
		return batch

def get_data_reg(data_path, file, pretrained_model, seed, augmentation_config=None):
	## Define n-gram LM file
	lm_text = data_path + 'random/' + '/' + seed + '/lm_text.txt'
	lm = data_path + 'random/' + '/' + seed + '/lm.arpa'
	lm_vocab = data_path + 'random/' + '/' + seed + '/lm_vocab.txt'

	data_path_file = data_path + file
	data_path_file = data_path_file.replace(u'\xa0', u'')
	original_data = pd.read_csv(data_path_file)
	path_list = original_data['path'].tolist()
	transcript_list = original_data['transcript'].tolist()

	# Initialize augmentation
	augmenter = AudioAugmentation()

	## Getting audio data
	data = []
	duration = 0
	words = []
	cleaned_transcripts = []

	for i in range(len(path_list)):
		wav_path = path_list[i]
		transcript = clean_sent(transcript_list[i])
		cleaned_transcripts.append(transcript)
		signal, samplerate = librosa.load(wav_path, sr=16000)
		
		if len(signal.shape) > 1 and signal.shape[1] == 2:
			signal = np.average(signal, axis=1)

		# Original sample
		entry = {}
		duration += len(signal) / samplerate
		words = words + transcript.split()
		entry["sentence"] = transcript.replace("\n", " ")				
		entry["audio"] = {"sampling_rate" : samplerate, "array" : signal}
		data.append(entry)

		# Add augmented samples for training data
		if 'train' in file and augmentation_config:
			try:
				augmented_signal = augmenter.apply_augmentation(signal, augmentation_config)
				aug_entry = {}
				aug_entry["sentence"] = transcript.replace("\n", " ")
				aug_entry["audio"] = {"sampling_rate" : samplerate, "array" : augmented_signal}
				data.append(aug_entry)
				duration += len(augmented_signal) / samplerate
			except Exception as e:
				print(f"Augmentation failed for sample {i}: {e}")

	print(f"Total duration: {duration:.2f} seconds")
	print(f"Total samples: {len(data)}")

	if 'train' in file:
		## Getting text data for training n-gram LM
		print('\nBUILDING LANGUAGE MODEL')
		print(f"LM files: {lm}, {lm_text}")
	
		with open(lm_text, 'w', encoding="utf-8") as f:
			for tok in cleaned_transcripts:
				f.write(tok + '\n')

		## Training an n-gram LM with SRILM
		os.system('module load gcc')
		os.system('module load perl/5.20.0')
		os.system(f'/blue/liu.ying/asr_resource/kaldi/tools/srilm/bin/i686-m64/ngram-count -order 3 -unk -write-vocab {lm_vocab} -wbdiscount -text {lm_text} -lm {lm}')

	return data, lm

def train(data_path, train_data, test_data, pretrained_model, seed, config_name="default"):
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

	with open(data_path + 'random/' + seed + '/vocab.json', 'w', encoding="utf-8") as vocab_file:
		json.dump(vocab_dict, vocab_file, ensure_ascii=False)
	
	## Creation of the tokeniser
	print("Setting up tokenizer...")
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(data_path + 'random/' + seed + '/', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	
	repo_name = f'asr_model/random/{seed}/{pretrained_model}_{config_name}/'
	print(f"Model will be saved to: {repo_name}")
	
	if not os.path.exists(repo_name):
		os.makedirs(repo_name)
		
	tokenizer.save_pretrained(repo_name)

	## Extraction of speech features
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
	
	# Build decoder - use processor vocabulary to match model output
	vocab = processor.tokenizer.get_vocab()
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
		
		# pyctcdecode
		logits_np = logits.cpu().numpy()[0]
		pyctc_pred = decoder.decode(logits_np)
		pyctc_preds.append(pyctc_pred)
	
	# Calculate metrics
	wer_metric = evaluate.load("wer")
	cer_metric = evaluate.load("cer")
	
	# pyctcdecode results
	pyctc_wer = wer_metric.compute(predictions=pyctc_preds, references=sentences)
	pyctc_cer = cer_metric.compute(predictions=pyctc_preds, references=sentences)
	
	# Print results
	print(f"\nPyctcdecode ({'with LM' if use_lm else 'no LM'}):")
	print(f"  WER: {pyctc_wer:.4f}")
	print(f"  CER: {pyctc_cer:.4f}")
	
	# Show some examples
	print("\nSAMPLE PREDICTIONS (first 3):")
	for i in range(min(3, len(sentences))):
		print(f"\nSample {i+1}:")
		print(f"  Ground truth: '{sentences[i]}'")
		print(f"  Pyctcdecode:  '{pyctc_preds[i]}'")
	
	# Save results
	results_file = os.path.join(model_path, "evaluation_results.txt")
	with open(results_file, 'w') as f:
		f.write(f"Pyctcdecode WER: {pyctc_wer:.4f}\n")
		f.write(f"Pyctcdecode CER: {pyctc_cer:.4f}\n")
		f.write(f"Language model used: {use_lm}\n")
	
	return pyctc_wer, pyctc_cer

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type=str, default="asr_model/")
	parser.add_argument("--seed", type=str, default="1")
	parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-xlsr-53")
	parser.add_argument("--skip_training", action="store_true", help="Skip training and only run evaluation")
	parser.add_argument("--compare_augmentations", action="store_true", help="Compare different augmentation strategies")
	parser.add_argument("--augmentation_config", type=str, default="medium_aug", 
	                   help="Augmentation configuration: no_aug, light_aug, medium_aug, heavy_aug, robustness_aug, speed_only, masking_only")

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

	if args.compare_augmentations:
		# Compare multiple augmentation configurations
		comparison = AugmentationComparison()
		configs = comparison.define_augmentation_configs()
		
		print(f"Comparing {len(configs)} augmentation configurations...")
		
		for config_name, config in configs.items():
			print(f"\n{'='*60}")
			print(f"TRAINING WITH CONFIGURATION: {config_name}")
			print(f"Config: {config}")
			print(f"{'='*60}")
			
			# Load data with current augmentation config
			print('Loading data...')
			train_data, lm_path = get_data_reg(data_path, f'sentence_train_{seed}.csv', pretrained_model, seed, config if config_name != 'no_aug' else None)
			test_data, _ = get_data_reg(data_path, f'sentence_test_{seed}.csv', pretrained_model, seed, None)  # Never augment test data
			
			print(f"Train samples: {len(train_data)}")
			print(f"Test samples: {len(test_data)}")
			
			# Train the model
			train_data, test_data, repo_name, vocab_dict, processor = train(data_path, train_data, test_data, pretrained_model, seed, config_name)
			
			# Evaluate with pyctcdecode
			wer, cer = evaluate_with_pyctcdecode(repo_name, test_data, vocab_dict, lm_path)
			
			# Save results
			comparison.save_results(config_name, wer, cer, repo_name)
			
			print(f"Configuration {config_name} completed - WER: {wer:.4f}, CER: {cer:.4f}")
		
		# Print final comparison
		comparison.print_comparison()
		
	else:
		# Single configuration training
		comparison = AugmentationComparison()
		configs = comparison.define_augmentation_configs()
		
		config = configs.get(args.augmentation_config, {})
		
		print('Loading data...')
		train_data, lm_path = get_data_reg(data_path, f'sentence_train_{seed}.csv', pretrained_model, seed, config if args.augmentation_config != 'no_aug' else None)
		test_data, _ = get_data_reg(data_path, f'sentence_test_{seed}.csv', pretrained_model, seed, None)  # Never augment test data

		print(f"Train samples: {len(train_data)}")
		print(f"Test samples: {len(test_data)}")
		print(f"Sample audio length: {len(train_data[0]['audio']['array'])}")
		print(f"Sample transcript: {train_data[0]['sentence']}")

		model_path = f'asr_model/random/{seed}/{pretrained_model}_{args.augmentation_config}'

		if not args.skip_training:
			# Train the model
			train_data, test_data, repo_name, vocab_dict, processor = train(data_path, train_data, test_data, pretrained_model, seed, args.augmentation_config)
			model_path = repo_name
		else:
			# Load existing vocab for evaluation only
			with open('vocab.json', 'r') as f:
				vocab_dict = json.load(f)
			print("Skipping training, loading existing model...")

		# Evaluate with pyctcdecode
		wer, cer = evaluate_with_pyctcdecode(model_path, test_data, vocab_dict, lm_path)
		
		print(f"\nFINAL SUMMARY:")
		print(f"Configuration: {args.augmentation_config}")
		print(f"Pyctcdecode WER: {wer:.4f}")
		print(f"Pyctcdecode CER: {cer:.4f}")

if __name__ == "__main__":
	main()
	print("Script finished running")