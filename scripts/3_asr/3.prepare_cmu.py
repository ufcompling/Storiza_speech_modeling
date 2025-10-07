import io, os
#import librosa
import pandas as pd

import re

arpabet_to_ipa_map = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ər',
    'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
    # Stress markers (often represented as diacritics in IPA, but can be handled separately)
    '0': '', '1': 'ˈ', '2': 'ˌ' 
}

def convert_arpabet_to_ipa(arpabet_sequence):
    ipa_result = []
    for phone in arpabet_sequence:
        # Handle stress markers
        if phone and phone[-1].isdigit():
            stress_marker = arpabet_to_ipa_map.get(phone[-1], '')
            base_phone = phone[:-1]
            ipa_result.append(stress_marker + arpabet_to_ipa_map.get(base_phone, phone))
        else:
            ipa_result.append(arpabet_to_ipa_map.get(phone, phone))
    return ''.join(ipa_result)

def convert_phonetic_transcription(text):
    def process_phonetic_match(match):
        # Extract the phonetic content between slashes
        phonetic_content = match.group(1)
        
        # Split by spaces to get individual ARPAbet phones
        arpabet_phones = phonetic_content.split()
        
        # Convert to IPA
        ipa_transcription = convert_arpabet_to_ipa(arpabet_phones)
        
        # Return with slashes
        return f'/{ipa_transcription}/'
    
    # Find all phonetic transcriptions and convert them
    pattern = r'/([^/]+)/'
    return re.sub(pattern, process_phonetic_match, text)

data_path = '/blue/liu.ying/Storiza_speech_modeling/cmu_kids/kids/'
audio_list = []
transcript_list = []
num_kid = 0
for kid in os.listdir(data_path):
    audio_path = data_path + kid + '/signal/'
    transcript_path = data_path + kid + '/trans/'
    if os.path.exists(audio_path) and os.path.exists(transcript_path):
        num_kid += 1
        for file in os.listdir(audio_path):
            if file.endswith('.sph'):
                audio_list.append(audio_path + file)
                with open(transcript_path + file.replace('.sph', '.trn'), 'r') as f:
                    line = f.read().strip().split()
                    transcript = ' '.join([tok for tok in line if '[' not in tok])
                    transcript = convert_phonetic_transcription(transcript)
                    transcript = transcript.replace('/', '')
                    transcript_list.append(transcript)
    else:
        pass

cmu_data = pd.DataFrame({'path': audio_list, 'transcript': transcript_list})
cmu_data.to_csv('/blue/liu.ying/Storiza_speech_modeling/asr_model/cmu_kids.csv', index=False)
print(f'Number of kids: {num_kid}') # N= 41