## Identify potentially problematic IP
import pandas as pd

target = "β" #'ɒ' #'ɐ'
data = pd.read_csv('processed_annotations/word_level_data.csv')
path_list = data['Path'].tolist()
IPA_list = data['IPA'].tolist()
for i, IPA in enumerate(IPA_list):
    try:
        if target in IPA:
            print(f"Found '{target}' in file: {path_list[i]} with IPA: {IPA}")
    except:
        pass