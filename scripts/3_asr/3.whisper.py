import numpy as np
import re, os
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperTokenizer, WhisperFeatureExtractor
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import soundfile as sf
import argparse
from tqdm import tqdm
from random import shuffle
import pandas as pd
import librosa

if not os.path.exists('asr_model/'):
    os.system('mkdir asr_model/')

def read_audio(fname):
    """Load an audio file and return PCM along with the sample rate"""
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

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper that pads inputs and labels
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_data_reg(data_path, file, pretrained_model, seed, cmu_audio=False, childes_lm=False, cmu_lm=False):
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
            entry["audio"] = {"sampling_rate": samplerate, "array": signal}
            entry["duration"] = duration
            data.append(entry)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue

    data = sorted(data, key=lambda entry: entry['duration'])
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Number of samples: {len(data)}")

    return data

def train(data_path, train_data, test_data, pretrained_model, seed):
    print("Setting up Whisper model and processor...")
    
    repo_name = f'asr_model/random/{seed}/{pretrained_model}/'
    print(f"Model will be saved to: {repo_name}")
    
    if not os.path.exists(repo_name):
        os.makedirs(repo_name)

    # Load Whisper processor (includes both feature extractor and tokenizer)
    processor = WhisperProcessor.from_pretrained(pretrained_model)
    
    # Whisper uses a different approach - it has a built-in multilingual tokenizer
    # We'll use English language setting
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def prepare_dataset(batch):
        audio = batch["audio"]
        
        # Compute log-Mel spectrogram input features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        
        return batch

    # Setting up data for training
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
    )
    wer_metric = evaluate.load("wer")
    
    train_data_temp = list(map(prepare_dataset, train_data))
    train_data_processed = [tok for tok in train_data_temp if tok is not None]
    test_data_temp = list(map(prepare_dataset, test_data))
    test_data_processed = [tok for tok in test_data_temp if tok is not None]

    print(f"Prepared train samples: {len(train_data_processed)}")
    print(f"Prepared test samples: {len(test_data_processed)}")
 
    print("Preparing Whisper model...")
    # Load pre-trained Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
    
    # Configure model for fine-tuning
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # IMPORTANT: Disable use_cache when using gradient checkpointing
    model.config.use_cache = False
    
    # Language-specific configuration
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    
    # Ensure model is in training mode
    model.train()
    
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    epochs = 30
    batch_size = 8

    training_args = Seq2SeqTrainingArguments(
        output_dir=repo_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix for backward through graph error
        fp16=True,
        eval_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
    )

    print("Starting training...")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data_processed,
        eval_dataset=test_data_processed,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    print("Training completed!")

    return train_data, test_data, repo_name, processor

def evaluate_model(model_path, test_data, processor=None):
    """
    Evaluate the trained Whisper model
    """
    print("\n" + "="*50)
    print("EVALUATING WHISPER MODEL")
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
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path).to("cuda")
    
    # Load or use provided processor
    if processor is None:
        # Try to load from checkpoint, fallback to original model name
        try:
            processor = WhisperProcessor.from_pretrained(checkpoint_path)
        except:
            # Extract original model name from path if checkpoint doesn't have processor
            print("Processor not found in checkpoint, loading from base model...")
            # Assume the model path contains the original model name
            if "whisper" in checkpoint_path:
                base_model = "openai/whisper-small"  # default fallback
                for size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]:
                    if size in checkpoint_path:
                        base_model = f"openai/whisper-{size}"
                        break
                processor = WhisperProcessor.from_pretrained(base_model)
            else:
                processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # Set language for generation
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    
    # Prepare data for evaluation
    signals = [x["audio"]["array"] for x in test_data]
    sentences = [x["sentence"] for x in test_data]
    
    print(f"Evaluating {len(signals)} samples...")
    
    predictions = []
    
    model.eval()
    for i in tqdm(range(len(signals))):
        signal = signals[i:i+1]
        
        # Handle stereo audio
        if len(signal[0].shape) > 1 and signal[0].shape[1] == 2:
            signal = [np.average(signal[0], axis=1)]
        
        # Prepare input features
        input_features = processor.feature_extractor(
            signal[0], 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to("cuda")
        
        # Generate predictions
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        
        # Decode predictions
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        predictions.append(transcription)
    
    # Calculate metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    wer = wer_metric.compute(predictions=predictions, references=sentences)
    cer = cer_metric.compute(predictions=predictions, references=sentences)
    
    # Print results
    print("\nEVALUATION RESULTS:")
    print(f"  WER: {wer:.4f}")
    print(f"  CER: {cer:.4f}")
    
    # Show some examples
    print("\nSAMPLE PREDICTIONS (first 3):")
    for i in range(min(3, len(sentences))):
        print(f"\nSample {i+1}:")
        print(f"  Ground truth: '{sentences[i]}'")
        print(f"  Prediction:   '{predictions[i]}'")
    
    # Save predictions
    predictions_file = os.path.join(model_path, "predictions.txt")
    with open(predictions_file, 'w') as f:
        for i in range(len(sentences)):
            f.write(f"Ground truth: {sentences[i]}\n")
            f.write(f"Prediction:   {predictions[i]}\n\n")

    # Save results
    results_file = os.path.join(model_path, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"WER: {wer:.4f}\n")
        f.write(f"CER: {cer:.4f}\n")
    
    return wer

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="asr_model/")
    parser.add_argument("--seed", type=str, default="1")
    parser.add_argument("--pretrained_model", type=str, default="openai/whisper-small", 
                       help="Whisper model size: openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large-v3")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--cmu_audio", action="store_true", help="Include CMU Kids audio data for training")
    parser.add_argument("--childes_lm", action="store_true", help="Include CHILDES data (not used in Whisper)")
    parser.add_argument("--cmu_lm", action="store_true", help="Include CMU Kids data (not used in Whisper)")
    args = parser.parse_args()

    data_path = args.data_path
    seed = args.seed
    pretrained_model = args.pretrained_model

    # Create directories
    for dir_path in [f'{data_path}random/', f'{data_path}random/{seed}']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    print('Loading data...')
    train_data = get_data_reg(data_path, f'sentence_train_{seed}.csv', pretrained_model, seed, args.cmu_audio)
    test_data = get_data_reg(data_path, f'sentence_test_{seed}.csv', pretrained_model, seed)

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Sample audio length: {len(train_data[0]['audio']['array'])}")
    print(f"Sample transcript: {train_data[0]['sentence']}")

    model_path = f'asr_model/random/{seed}/{pretrained_model.replace("/", "_")}'

    if not args.skip_training:
        # Train the model
        train_data, test_data, repo_name, processor = train(data_path, train_data, test_data, pretrained_model, seed)
        model_path = repo_name
    else:
        # Load existing processor for evaluation only
        processor = WhisperProcessor.from_pretrained(model_path)
        print("Skipping training, loading existing model...")

    # Evaluate the model
    wer = evaluate_model(model_path, test_data, processor)
    
    print(f"\nFINAL SUMMARY:")
    print(f"Test WER: {wer:.4f}")

if __name__ == "__main__":
    main()
    print("Script finished running")
