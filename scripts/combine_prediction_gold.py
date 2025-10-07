import io, os
import pandas as pd 

data = pd.read_csv('error_category_prediction_confidence_data.csv')
path_list = data['Path'].tolist()
true_label_list = []

word_segments_data_file = 'processed_annotations/word_level_data_ngram.csv'
full_word_segments_data = pd.read_csv(word_segments_data_file)

word_segments_data = full_word_segments_data.sample(frac=1)
audio_path_list = word_segments_data['Path'].tolist()

word_segments_data["Error Category"] = word_segments_data["Error Category"].apply(
    lambda x: [category.strip() for category in x.split("+") if category != 'Mixed Error']
)
error_category_list = word_segments_data['Error Category'].tolist()

path_label_dict = {}
for i in range(len(word_segments_data)):
	path_label_dict[audio_path_list[i]] = error_category_list[i]

for i in range(len(path_list)):
	label = ''
	if 'Correct' in path_label_dict[path_list[i]]:
		label = 1
	else:
		label = 0
	true_label_list.append(label)

data['True_label'] = true_label_list
data.to_csv('confidence_data.csv', index=False)
