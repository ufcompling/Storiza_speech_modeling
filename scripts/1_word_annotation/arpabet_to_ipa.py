### Combine data from cmu dictionary and English Wikitionary
### Convert ARPABET to IPA

import io, os
import json

CMUDICT_PATH = 'processed_annotations/cmudict-0.7b'
WIKTIONARY_PATH = 'processed_annotations/Wiktionary_arpabet.tsv'


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

def convert_cmu(CMUDICT_PATH):
	data = {}
	with open(CMUDICT_PATH, encoding = 'utf-8') as f:
		for line in f:
			if not line.startswith(';;;'):
				toks = line.strip().split()
				word = toks[0].lower()
				arpabet_sequence = toks[1 : ]
				ipa_sequence = convert_arpabet_to_ipa(arpabet_sequence)
				data[word] = ipa_sequence
				
#	for k, v in data.items():
#		print(k, v)
	return data

def convert_wiktionary(WIKTIONARY_PATH):
	data = {}
	with open(WIKTIONARY_PATH, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')
			word = toks[0].lower()
			arpabet_sequence = toks[1].split()
			ipa_sequence = convert_arpabet_to_ipa(arpabet_sequence)
			data[word] = ipa_sequence
	return data

full_dict = convert_cmu(CMUDICT_PATH)
full_dict.update(convert_wiktionary(WIKTIONARY_PATH))

## Add the following words from Storiza corpus that are not in the dictionary
additional = {
	"thudded": "θʌdɪd",
	"thuds": "θʌdz",
	"Zade": "zeɪd",
	"Zade's": "zeɪdz",
	"pranced": "prænst",
	"Luna's": "lunəz",
	"Whistolot": "wɪstəlɑt",
	"Whizzy": "wɪzi",
	"Whisperberry": "wɪspəɹbɛɹi",
	"Whistolots": "wɪstəlɑts",
	"'th'": "θ",
	"cubing": "kjubɪŋ",
	"Ssssss": "s:",
	"Starshine": "stɑɹʃaɪn",
	"chirped": "tʃɜɹpt",
	"swishing": "swɪʃɪŋ",
	"cube's": "kjubz",
	"GymSoccer": "dʒɪmsɑkəɹ",
	"Catsy": "kætsi",
	"racecourse": "reɪskɔɹs",
	"Lakeville": "leɪkvɪl",
	"lounged": "laʊndʒd",
	"what is": "wʌt ɪz",
	"E.A.": "ieɪ",
	"can a": "kæn ə", 
	"playdough": "pleɪdoʊ",
	"messiest": "mɛsiɪst",
	"whooshed": "wuʃt",
	"Timmy's": "tɪmiz",
	"trinklets": "trɪŋklɪts",
	"was it": "wʌz ɪt",
	"fetcher": "fɛtʃəɹ",
	"cheetah's": "tʃitəz",
	"Gil's": "ɡɪlz",
	"unicorns": "junɪkɔɹnz",
	"squealed": "skwild",
	"I don't": "aɪ doʊnt",
	"croaked": "kroʊkt",
	"purred": "pɜɹd",
	"waterfall's": "wɔtəɹfɔlz",
	"Jill's": "dʒɪlz",
	"Dums": "dʌmz",
	"tastiest": "teɪstiɪst",
	"meowed": "miaʊd",
	"whizzes": "wɪzɪz",
	"I mean": "aɪ min",
	"2": "tu",
	"DLC": "diɛlsi/",
	"go-ahead": "goʊəhɛd",
	"itskapt": "ɪtskæpt",
	"flowerbeds": "flaʊəɹbɛdz",
	"stuffies": "stʌfiz",
	"picnicker": "pɪknɪkəɹ",
	"Sunny's": "sʌniz",
	"Kitty's": "kɪtiz",
	"around the": "əɹaʊnd ðə",
	"Gabi": "ɡæbi",
	"mermaid's": "mɜɹmeɪdz",
	"chinchillas": "tʃɪntʃɪləz",
	"chirped": "tʃɜɹpt",
	"Pokemon": "poʊkimɑn",
	"snored": "snɔɹd",
	"kitties": "kɪtiz",
	"There's": "ðɛɹz",
	"Eevee": "ivi",
	"Pokeball": "poʊkibɔl",
	"Jigglypuff": "dʒɪɡlipʌf",
	"on a": "ɑn ə",
	"Melia's": "miliəz",
	"blindfolds": "blaɪndfoʊldz",
	"swishing": "swɪʃɪŋ",
	"puppy's": "pʌpiz",
	"Ferdo's": "fɜɹdoʊz",
	"Ferdo": "fɜɹdoʊ",
	"purred": "pɜɹd",
	"Paintville": "peɪntvɪl",
	"preheated": "pɹihitɪd",
	"preplan": "pɹiˈplæn",
	"Bree's": "bɹiz",
	"mercat": "mɜɹkæt",
	"I mean": "aɪ min",
	"lamp's": "læmps",
	"ows": "ows"
}

full_dict.update(additional)

with open("processed_annotations/full_en_dict.json", "w") as f:
	json.dump(full_dict, f, indent = 4)
