import io, os
import pandas as pd
import string

punctuations = list(string.punctuation)

sentence_level_data = pd.read_csv('processed_annotations/sentence_level_data.csv')

goldstandard_list = sentence_level_data['goldStandard'].tolist()
intended_words_list = sentence_level_data['Intended Words'].tolist()

all_omitted_words = []
total_words = 0

for i in range(len(goldstandard_list)):
	omitted_words = []
	goldstandard = goldstandard_list[i].lower()
	for punct in punctuations:
		if punct in goldstandard:
			goldstandard = goldstandard.replace(punct, '')
	goldstandard = goldstandard.split()
	total_words += len(goldstandard)
	intended_words = intended_words_list[i].lower()
	for punct in punctuations:
		if punct in intended_words:
			intended_words = intended_words.replace(punct, '')
	intended_words = intended_words.split()
	for z in range(len(goldstandard)):
		word = goldstandard[z]
		if word not in intended_words:
			omitted_words.append(word)
	all_omitted_words += omitted_words
	if omitted_words != []:
		print(goldstandard)
		print(intended_words)
		print(omitted_words)
		print('\n')

print('Number of omitted words in total:', len(all_omitted_words))
print('Number of words in total:', total_words)
print('Proportion of omitted words:', round(100 * len(all_omitted_words)/total_words, 2))