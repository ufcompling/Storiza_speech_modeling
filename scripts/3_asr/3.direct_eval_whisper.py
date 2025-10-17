import whisper
import io, os
import pandas as pd 
#from jiwer import wer
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

os.makedirs('whisper_models', exist_ok=True)
os.makedirs('whisper_models/pretrained', exist_ok=True)

sentence_level_data = pd.read_csv('processed_annotations/sentence_level_data.csv')
audio_path_list = sentence_level_data['Path']
transcript_list = sentence_level_data['Transcript']

whisper_models = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']

for model_option in whisper_models:
	print(model_option)
	model_transcript_list = []
	for i in range(len(sentence_level_data)):
		audio = audio_path_list[i]
		transcript = transcript_list[i]
		model = whisper.load_model(model_option)
		result = model.transcribe(audio)
		model_transcript = result["text"]
		model_transcript_list.append(model_transcript)

	# Processor results (should match training eval)
	wer = wer_metric.compute(predictions=model_transcript_list, references=transcript_list)
	cer = cer_metric.compute(predictions=model_transcript_list, references=transcript_list)

	with open('whisper_models/pretrained/' + model_option + '_predictions.txt', 'w') as f:
		f.write(str(wer) + '\n')
		f.write(str(cer) + '\n')
		for transcript in model_transcript_list:
			f.write(str(transcript) + '\n')

	f.close()

	print('Evaluation done for ' + model_option)
