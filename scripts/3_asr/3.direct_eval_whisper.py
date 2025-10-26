import whisper
import io, os, sys
import pandas as pd 
#from jiwer import wer
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

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

os.makedirs('asr_model/pretrained_eval/openai', exist_ok=True)

sentence_level_data = pd.read_csv('processed_annotations/sentence_level_data.csv')
audio_path_list = sentence_level_data['Path']
original_transcript_list = sentence_level_data['Transcript'].tolist()
transcript_list = [strip_punctuation_except_apostrophe(transc) for transc in original_transcript_list]
transcript_list = [transc.lower() for transc in transcript_list]

whisper_models = [sys.argv[1]] # ['tiny', 'base', 'small', 'medium', 'large', 'turbo']

for model_option in whisper_models:
	print(model_option)
	os.makedirs('asr_model/pretrained_eval/openai/' + 'whisper_' + model_option, exist_ok=True)
	model_transcript_list = []
	for i in range(len(audio_path_list)):
		audio = audio_path_list[i]
		model = whisper.load_model(model_option)
		result = model.transcribe(audio)
		model_transcript = result["text"]
		model_transcript_list.append(model_transcript)

	original_wer = wer_metric.compute(predictions=model_transcript_list, references=original_transcript_list)
	original_cer = cer_metric.compute(predictions=model_transcript_list, references=original_transcript_list)

	lower_model_transcript_list = [strip_punctuation_except_apostrophe(transc).lower() for transc in model_transcript_list]
	strip_wer = wer_metric.compute(predictions=lower_model_transcript_list, references=transcript_list)
	strip_cer = cer_metric.compute(predictions=lower_model_transcript_list, references=transcript_list)

	with open('asr_model/pretrained_eval/openai/' + 'whisper_' + model_option + '/evaluation_results.txt', 'w') as f:
		f.write('Original WER' + '\n')
		f.write(str(original_wer) + '\n')
		f.write(str(original_cer) + '\n')
		f.write('\n')
		f.write('Lower+Stripping WER' + '\n')
		f.write(str(strip_wer) + '\n')
		f.write(str(strip_cer) + '\n')

	f.close()
	
	with open('asr_model/pretrained_eval/openai/' + 'whisper_' + model_option + '/predictions.txt', 'w') as f:
		for transcript in model_transcript_list:
			f.write(str(transcript) + '\n')

	f.close()

	print('Evaluation done for ' + model_option)
