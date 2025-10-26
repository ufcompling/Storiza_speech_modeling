import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import pandas as pd
import evaluate

import string

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

# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
model_path, 
	device_map="cuda", 
	torch_dtype="auto", 
	trust_remote_code=True,
	# if you do not use Ampere or later GPUs, change attention to "eager"
	_attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
speech_prompt = "Transcribe the audio to text, and then translate the audio to French. The speech was produced by children in elementary school"
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
print(f'Prompt\n{prompt}')

### Generating responses and evaluation

# Calculate metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

response_list = []
for file in audio_list:
	audio, samplerate = sf.read(file)
	inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')
	generate_ids = model.generate(
		**inputs,
		max_new_tokens=1000,
		generation_config=generation_config,
	)
	generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
	response = processor.batch_decode(
		generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0]
	response_list.append(response)

original_wer = wer_metric.compute(predictions=response_list, references=original_transcript_list)
original_cer = cer_metric.compute(predictions=response_list, references=original_transcript_list)

lower_response_list = [strip_punctuation_except_apostrophe(transc).lower() for transc in response_list]
strip_wer = wer_metric.compute(predictions=lower_response_list, references=transcript_list)
strip_cer = cer_metric.compute(predictions=lower_response_list, references=transcript_list)

os.makedirs('asr_model/pretrained_eval/phi4', exist_ok=True)

with open('asr_model/pretrained_eval/phi4/evaluation_results.txt', 'w') as f:
		f.write('Original WER' + '\n')
		f.write(str(original_wer) + '\n')
		f.write(str(original_cer) + '\n')
		f.write('\n')
		f.write('Lower+Stripping WER' + '\n')
		f.write(str(strip_wer) + '\n')
		f.write(str(strip_cer) + '\n')

f.close()
	
with open('asr_model/pretrained_eval/phi4/predictions.txt', 'w') as f:
	for transcript in response_list:
		f.write(str(transcript) + '\n')

f.close()

