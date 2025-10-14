## Check how much of the production is real words or not

import io, os
import pandas as pd
import string
import json

punctuations = list(string.punctuation)

sentence_level_data = pd.read_csv('processed_annotations/sentence_level_data.csv')

EN_IPA_DICT = "processed_annotations/full_en_dict.json"
with open(EN_IPA_DICT, "r", encoding = "utf-8") as f:
	en_ipa_dict = json.load(f)

goldstandard_list = sentence_level_data['goldStandard'].tolist()
transcript_list = sentence_level_data['Transcript'].tolist()

all_nounce_words = []
total_produced_words = 0

for i in range(len(transcript_list)):
	nounce_words = []
	transcript = transcript_list[i].lower()
	for punct in punctuations:
		if punct in transcript:
			transcript = transcript.replace(punct, '')
	transcript = transcript.split()
	total_produced_words += len(transcript)
	for z in range(len(transcript)):
		word = transcript[z]
		if word not in en_ipa_dict:
			nounce_words.append(word)
	all_nounce_words += nounce_words

print('Nounce words:')
for word in list(set(all_nounce_words)):
	print(word)

print('\n')
print('Number of nounce words in total:', len(all_nounce_words))
print('Number of produced words in total:', total_produced_words)
print('Proportion of nounce words:', round(100 * len(all_nounce_words)/total_produced_words, 2))
