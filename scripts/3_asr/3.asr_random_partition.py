## Given an input path, in which there is a wav folder and a txt folder
## Randomly partition the audio and their corresponding transcripts into a training and test set

## e.g., python scripts/random_partition.py hupa asr_corpora/ data/

import os, re, io
import soundfile as sf
import random
import itertools
import numpy as np
from tqdm import tqdm
from inspect import getouterframes, currentframe
import pandas as pd
import sys

if not os.path.exists('asr_model/'):
	os.system('mkdir asr_model/')

original_data = pd.read_csv(sys.argv[1])
level = sys.argv[1].split('/')[1].split('_')[0]
data_path_list = original_data['Path'].tolist()
data_transcript_list = original_data['Transcript'].tolist()
data_goldStandard_list = original_data['goldStandard'].tolist()

output_path = 'asr_model/' 
n_random_splits = 1


data = []
total_dur = 0

for i in range(len(data_path_list)):
	file = data_path_list[i]
	transcript = data_transcript_list[i].strip()
	if file.endswith('wav'):
		sr = 16000
		signal, sr = sf.read(file) # signal and sampling rate
		dur = len(signal) / sr # audio duration		

		# Only including audio that is at least 5s
		# wav2vec does not handle short audio very well
		if dur >= 5 and len(transcript.split()) > 1:
			total_dur += dur
			data.append([file, transcript, dur])
		else:
			pass

for i in range(0, n_random_splits):
	i += 1
	i = str(i)
	random.shuffle(data)

	train_dur = 0
	test_dur = 0
	train_data = []
	test_data = []

	for tok in data:

		## Splitting the full corpus into training and test at a 4:1 ratio
		if train_dur <= total_dur * 0.8:
			train_data.append(tok)
			train_dur += tok[-1]
		else:
			test_data.append(tok)
			test_dur += tok[-1]

	# Up until this point, train_dur is in seconds
	# Converting to hours
	train_dur = train_dur / 3600
	test_dur = test_dur / 3600

	train_h = int(train_dur)
	train_min = int((train_dur - train_h) * 60)
	test_h = int(test_dur)
	test_min = int((test_dur - test_h) * 60)

	with open('storiza_asr_descriptive.txt', 'w') as f:
		f.write('train duration\ttest duration' + '\n')
		f.write(str(train_h) + 'h' + str(train_min) + 'min\t' + str(test_h) + 'h' + str(test_min) + 'min' + '\n')
	print('train duration: ' + str(train_h) + 'h' + str(train_min) + 'min')
	print('test duration: ' + str(test_h) + 'h' + str(test_min) + 'min')
	print('')

	train_wav_path_list = [tok[0] for tok in train_data]
	train_transcript_list = [tok[1] for tok in train_data]
	train_dur_list = [tok[-1] for tok in train_data]

	train_output = pd.DataFrame({'path': train_wav_path_list,
			  	'transcript': train_transcript_list,
			  	'duration': train_dur_list})

	train_output.to_csv(output_path + level + '_train_' + str(i) + '.csv', index = False)

	test_wav_path_list = [tok[0] for tok in test_data]
	test_transcript_list = [tok[1] for tok in test_data]
	test_dur_list = [tok[-1] for tok in test_data]

	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list,
			  	'duration': test_dur_list})

	test_output.to_csv(output_path + level + '_test_' + str(i) + '.csv', index = False)

