import io, os

### reading in sentences in CoNLL format ###

def conll_read_sentence(file_handle):

	sent = []

	for line in file_handle:
		line = line.strip('\n')

		if line.startswith('#') is False :
			toks = line.split("\t")

			if len(toks) == 1:
				return sent
			else:
				if toks[0].isdigit() == True:
					sent.append(toks)

	return None

def extract_annotations(file_handle):

    all_sents = []

    with io.open(file_handle, encoding = 'utf-8') as f:
        sent = conll_read_sentence(f)

        while sent is not None:
            all_sents.append(sent)
            sent = conll_read_sentence(f)
    
    return all_sents

all_child_transcripts = []
for file in os.listdir('../CHILDES_English/'):
	if file.endswith('.conllu'):
		file_annotations = extract_annotations('../CHILDES_English/' + file)
		for sent in file_annotations:
			if len(sent) >= 5:
				speaker = sent[0][8].split()[2]
				if 'CHILD' or 'Child' in speaker:
					transcript = [tok[1] for tok in sent]
					print(len(sent), transcript)
					all_child_transcripts.append(' '.join(transcript))

with open('scripts/3_asr/CHILDES_child_transcripts.txt', 'w') as f:
	for transcript in all_child_transcripts:
		f.write(transcript + '\n')
