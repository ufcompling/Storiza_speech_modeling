import re
from collections import defaultdict
import os, json
from functools import lru_cache
from typing import Dict, List, Tuple, Any
from pydub import AudioSegment
import statistics
import pandas as pd

sec_to_ms = 1000
website_prefix = 'https://2025storiza.michaelbennie.org/audio_clips/'

# Loading IPA dictionary constructed from CMU dictionary and Wiktionary
EN_IPA_DICT = "processed_annotations/full_en_dict.json"
with open(EN_IPA_DICT, "r", encoding = "utf-8") as f:
	en_ipa_dict = json.load(f)
 
# Loading information about user id of each child and their grade
STORY_META_CSV = os.path.join("processed_annotations/story.xlsx")
story_meta_data = pd.read_excel(STORY_META_CSV)
story_id_list = story_meta_data['__id__'].tolist()
grade_list = story_meta_data['grade'].tolist()
userID_list = story_meta_data['userId (matches the uid in the recording file name)'].tolist()
userID_grade_dict = {}
for i in range(len(userID_list)):
  userID_grade_dict[story_id_list[i] + ' ' + userID_list[i]] = grade_list[i]


# For mixed errors, we map the error category for consistency
error_map = {
    'Grammatical': 'Grammatical Error',
    'Orthographic Sub.': 'Orthographic Error',
    'Phonological': 'Phonological Error',
    'Run-on': 'Run-on Word',
    'Structural': 'Structural Error',
    'Visual Tracking': 'Visual Tracking Error',
    'Contraction/Shortening': 'Correct'
}

def storiza_URL_to_timestamps(url: str) -> list[float, float]:
	filename = url[len(website_prefix):]
	items = filename.split("_")
	if len(items) < 3:
		raise ValueError('URL illformed')
	return list(map(float, (items[0], items[2])))


# ----------------------------
# Helpers: audio loading/caching
# ----------------------------
class AudioCache:
	"""Caches pydub AudioSegment and duration per original_audio_name."""
	def __init__(self, base_dir: str):
		self.base_dir = base_dir
		self._audio: Dict[str, AudioSegment] = {}
		self._duration: Dict[str, float] = {}

	def get(self, original_audio_name: str) -> Tuple[AudioSegment, float]:
		if original_audio_name not in self._audio:
			path = os.path.join(self.base_dir, original_audio_name)
			audio = AudioSegment.from_file(path)
			self._audio[original_audio_name] = audio
			self._duration[original_audio_name] = audio.duration_seconds
		return self._audio[original_audio_name], self._duration[original_audio_name]


# ----------------------------
# Helpers: annotation parsing
# ----------------------------
def clean_ipa(ipa_list: List[str]) -> List[str]:
	"""
	Normalizes the first IPA string in `ipa_list`:
	  - strips slashes
	  - removes tie-bars in affricates (͡, ͜) and collapses to digraphs
	  - expands rhotized vowels (precomposed ɚ/ɝ and vowel + ˞) to vowel + ɹ
	  - converts "upwards r" variants to ɹ

	Returns a single-element list with the cleaned string (mirrors your original API).
	"""
	if not ipa_list:
		return ipa_list

	s = ipa_list[0]

	# 1) strip surrounding slashes if present
	if "/" in s:
		s = s.replace("/", "")
	if "[" in s:
		s = s.replace("[", "")
	if "]" in s:
		s = s.replace("]", "")
	if "\"" in s:
		s = s.replace("\"", "")

	# 2) remove tie bars so affricates become plain digraphs (e.g., t͡s -> ts, d͜z -> dz)
	#    U+0361 COMBINING DOUBLE INVERTED BREVE (͡), U+035C COMBINING DOUBLE BREVE BELOW (͜)
	s = s.replace("\u0361", "").replace("\u035C", "")

	# Keep explicit affricate simplifications too (no-op now, but harmless and explicit)
	s = s.replace("d͡ʒ", "dʒ").replace("t͡ʃ", "tʃ")

	# 3) RHOTICS: expand rhotized vowels to vowel + ɹ
	#    - precomposed: ɚ -> əɹ, ɝ -> ɜɹ (approximation that preserves rhoticity)
	s = s.replace("ɚ", "əɹ").replace("ɝ", "ɜɹ")

	#    - combining rhotic hook: any vowel + ˞ (U+02DE) -> vowel + ɹ
	#      We'll match a fairly broad set of IPA vowel symbols.
	IPA_VOWELS = "aeiou" \
				 "ɑæɐəɜɛeɪioɔuʊʌɒœøɯyɨʉɘɵɤʏɶɞ"

	# replace VOWEL + ˞ with VOWEL + ɹ
	s = re.sub(fr"([{IPA_VOWELS}])\u02DE", r"\1ɹ", s)

	#    - superscript r after a vowel: V + ʳ -> V + ɹ
	s = re.sub(fr"([{IPA_VOWELS}])\u02B3", r"\1ɹ", s)  # U+02B3 = ʳ

	# 4) Normalize all "upwards r" variants to ɹ
	#    (choose what you want included here; this set is intentionally broad)
	R_VARIANTS = [
		"r",   # alveolar trill
		"ɻ",   # retroflex approximant
		"ʳ",   # modifier letter small r (superscript)
		"ʴ",   # modifier letter small turned r with hook
	]
	for rv in R_VARIANTS:
		s = s.replace(rv, "ɹ")

	# 5) (optional) seperate any accidental double ɹɹ that could arise from replacements (rare)
	s = re.sub("ɹ{2,}", "ɹɹ.", s)

	# Consolidating prolongation
	if 'ː' in s:
		s = s.replace('ː', ':')

	if 'β' in s:
		s = s.replace('β', 'b')
	if 'ʋ' in s:
		s = s.replace('ʋ', 'ɹ')
	if 'ʑ' in s:
		s = s.replace('ʑ', 'ʒ')
	if 'ɫ' in s:
		s = s.replace('ɫ', 'l')
	if 'ɝ' in s:
		s = s.replace('ɝ', 'ɜɹ')

	# Diacritics
	if "ʰ" in s:
		s = s.replace("ʰ", '')
	if "ˈ" in s:
		s = s.replace("ˈ", '')
	if "ˌ" in s:
		s = s.replace("ˌ", '')
	if "ː" in s:
		s = s.replace("ː", '')
	if "̃" in s:
		s = s.replace("̃", '')
	if "̈" in s:
		s = s.replace("̈", '')
	if "̚" in s:
		s = s.replace("̚", '')
	if "̩" in s:
		s = s.replace("̩", '')
	if "̰" in s:
		s = s.replace("̰", '')
	if "̚" in s:
		s = s.replace("̚", '')

	# Changing some vowels
	if 'ɐ' in s:
		index_list = []
		for i in range(len(s)):
			phoneme = s[i]
			if phoneme == 'ɐ':
				index_list.append(i)
	
		# ɐ followed by ɪ --> a
		# ɐ followed by ʊ --> a
		# ɐ (elsewhere) --> ɑ
		s = list(s)
		follow_check = 0 
		for idx in index_list:
			try:
				next_phoneme = index_list[idx+1]
				if next_phoneme == 'ɪ' or next_phoneme == 'ʊ':
					print(s)
					s[idx] = 'a'
					follow_check += 1
			except:
				pass
		if follow_check == 0:
			for idx in index_list:
				s[idx] = 'ɑ'
		s = ' '.join(s)

	if 'ɒ' in s:
		s = s.replace('ɒ', 'ɑ')

	return [s]

def decide_actual_production(error_category: str, intended: str, produced: str, ipa: List[str]) -> str:
	if error_category == 'Correct': #bruh
		return intended, 'word'
	if intended and not produced and not ipa:
		return intended, 'word'
	if intended and not produced and ipa:
		return ipa[0], 'ipa'
	if intended and produced and not ipa:
		return produced, 'word'
	if intended and produced and ipa:
		return produced, 'word'
	if (not intended) and (not produced) and ipa:
		return ipa[0], 'ipa'
	if (not intended) and produced and not ipa:
		return produced, 'word'
	if (not intended) and produced and ipa:
		return produced, 'word'
	return '', ''


def parse_grouped_results(annotations_block: Dict[str, Any], goldStandard: str, original_audio_name: str) -> Dict[str, Dict[str, Any]]:
	"""
	Rebuilds the 'grouped' dict keyed by result_id with all fields populated.
	"""
	result = annotations_block.get("result", []) or []
	annotator = annotations_block.get("completed_by", {}).get("email")

	grouped = defaultdict(lambda: {
		"task_id": None,  # set by caller
		"annotator": annotator,
		"start": None,
		"end": None,
		"intended_words": [],
		"produced_word": [],
		"IPA": [],
		"original_error_category": '',
		"error_category": '',
		"error_labels": [],
		"comments": [],
		"goldStandard": goldStandard,
		"original_audio_name": original_audio_name,
		"actual_production": '',
		"actual_ipa": ''
	})

	for r in result:
		rid = r.get("id")
		value = r.get("value", {})
		r_type = r.get("type")
		from_name = r.get("from_name")

		if r_type == "labels" and from_name == "WordAnnotation":
			grouped[rid]["start"] = value.get("start")
			grouped[rid]["end"] = value.get("end")
			category = value.get("labels", [''])[0]
			grouped[rid]["error_category"] = category
			grouped[rid]["original_error_category"] = category

		if grouped[rid]["original_error_category"] != "Mixed Error" and r_type == "choices":
			category = grouped[rid]["error_category"]
			if category and category != "Mixed Error":
				category_no_space = category.replace(' ', '')
				if category_no_space in from_name:
					grouped[rid]["error_labels"].extend(value.get("choices", []))

		if r_type == "textarea" and from_name == "spoken_words":
			category = grouped[rid]["original_error_category"]
			if category in ["Run-on", "Contraction/Shortening"]:
				grouped[rid]["intended_words"].extend(value.get("text", []))

		if r_type == "textarea" and from_name == "spoken_word":
			category = grouped[rid]["original_error_category"]
			if category in [
				"Correct",
				"Phonological Error",
				"Orthographic Error",
				"Grammatical Error",
				"Structural Error",
				"Visual Tracking Error",
				"Disfluency Error",
				"Mixed Error",
			] or "Mixed Error+" in category:
				grouped[rid]["intended_words"].extend(value.get("text", []))

		if r_type == "textarea" and from_name == "produced_word":
			if grouped[rid]["original_error_category"] in [
				"Contraction/Shortening",
				"Visual Tracking Error",
				"Self Response",
				"Other",
				"Orthographic Error",
				"Grammatical Error",
				"Structural Error",
				"Mixed Error"
			]:
				grouped[rid]["produced_word"].extend(value.get("text", []))

		if r_type == "textarea" and from_name == "mispronunciation_word":
			if grouped[rid]["original_error_category"] in [
				"Contraction/Shortening",
				"Other",
				"Phonological Error",
				"Orthographic Error",
				"Grammatical Error",
				"Structural Error",
				"Visual Tracking Error",
				"Disfluency Error",
				"Mixed Error"
			]:
				grouped[rid]["IPA"].extend(value.get("text", []))

		if r_type == "textarea" and from_name == "issues":
			grouped[rid]["comments"].extend(value.get("text", []))

		if grouped[rid]["original_error_category"] == "Mixed Error" and r_type == "taxonomy":
			category = grouped[rid]["original_error_category"]
			labels = []
			mixed = value.get("taxonomy", [])
			for tok in mixed:
				if tok and tok[0] not in category:
					category = category + '+' + tok[0]
				if len(tok) > 1:
					labels += tok[1:]
			grouped[rid]["error_category"] = category
			grouped[rid]["error_labels"] = labels

	return grouped


def sorted_word_rows_from_grouped(
	task_id: int,
	grouped: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
	"""Converts grouped pieces into per-word rows with IPA cleaning and actual_production."""
	rows = []
	for _, info in grouped.items():
		if (
			info.get('error_labels') != ["Unfilled Pause"] and
			info.get('start') is not None and
			info.get('end') is not None
		):
			annotator = info.get("annotator")
			labels = info.get("error_labels", [])
			intended = ' '.join(info.get("intended_words", []))
			produced = ' '.join(info.get("produced_word", []))
			ipa = clean_ipa(info.get("IPA", []))

			try:
				if "Broken Word" in labels:
					if "rose" in annotator and "Stutter" not in labels:
						if '...' in ipa[0]:
							ipa[0] = ipa[0].replace('...', '|')
						elif '..' in ipa[0]:
							ipa[0] = ipa[0].replace('..', '|')
						elif '.' in ipa[0]:
							ipa[0] = ipa[0].replace('.', '|')

						if produced != '':
							if '...' in produced:
							  produced = produced.replace('...', '|')
							elif '..' in produced:
							  produced = produced.replace('..', '|')
							elif '.' in produced:
							  produced = produced.replace('.', '|')
			except:
				pass

			## Check if produced word has incorrect forms
			if "(" in produced:
				print('PRODUCED WORD has incorrect form:', produced, task_id, intended)
			if "unintelligible" in produced:
				print('PRODUCED WORD has incorrect form:', produced, task_id, intended)
			if " " in produced:
				print('PRODUCED WORD is more than one word:', produced, ipa, task_id, intended)

			## Check if IPA has incorrect forms
			if ipa != []:
				if "\"" in ipa[0]:
					print('IPA has incorrect form:', actual, ipa[0], task_id, intended)


			error_category = info.get("error_category", '')

			actual, form = decide_actual_production(error_category, intended, produced, ipa)

			actual_ipa = ''
			if form == 'ipa':
				actual_ipa = actual
			elif form != '':
				if ipa != []:
					actual_ipa = ipa[0]
				else:
					try:
						actual_ipa = en_ipa_dict[actual.lower()]
					except:
						try:
							actual_ipa = en_ipa_dict[actual]
						except:
							print("DID NOT FIND IPA FOR THIS WORD:", actual, task_id, intended, produced, ipa)

			rows.append({
				"task_id": task_id,
				"annotator": info.get("annotator"),
				"start": info.get("start"),
				"end": info.get("end"),
				"intended_words": intended,
				"produced_word": produced,
				"IPA": ipa,
				"error_category": error_category,
				"error_labels": info.get("error_labels", []),
				"comments": " | ".join(info.get("comments", [])),
				"goldStandard": info.get("goldStandard"),
				"original_audio_name": info.get("original_audio_name"),
				"actual_production": actual,
				"actual_ipa": actual_ipa
			})
	return sorted(rows, key=lambda x: (x["start"] is None, x["start"]))


def derive_sentence_window_from_words(
	word_rows: List[Dict[str, Any]],
	audio_duration_s: float,
	buffer_s: float = 0.010
) -> Tuple[float | None, float | None]:
	"""Replicates original clamping/buffering behavior."""
	valid_starts = [w["start"] for w in word_rows if w["start"] is not None]
	valid_ends = [w["end"] for w in word_rows if w["end"] is not None]
	if not valid_starts or not valid_ends:
		return None, None

	derived_start = max(0.0, min(valid_starts) - buffer_s)
	derived_end = min(audio_duration_s, max(valid_ends) + buffer_s)
	if derived_end <= derived_start:
		derived_start = max(0.0, min(valid_starts))
		derived_end = min(audio_duration_s, max(valid_ends))
		if derived_end <= derived_start:
			return None, None
	return derived_start, derived_end


def export_clip(audio: AudioSegment, start_s: float, end_s: float, out_path: str) -> None:
	start_ms = int(start_s * sec_to_ms)
	end_ms = int(end_s * sec_to_ms)
	audio[start_ms:end_ms].export(out_path, format='wav')


# ----------------------------
# Main function: same signature & outputs
# ----------------------------
def build_segments_and_rows(
	data,
	audio_base_dir,
	SENTENCE_SEGMENTS_DIR,
	WORD_SEGMENTS_DIR,
	WORD_SEGMENTS_NGRAM_DIR,
	BUFFERED_UTTERANCES_SUBDIR,  # kept for signature parity
	sentenceLabels_original_audio_timestamps,
	NGRAM,
	error_dict,
	sample_rate=4800,
	word_offset=1.0,
	debug=False
):
	"""
	Processes Label Studio JSON 'data' into sentence/word audio segments and metadata rows.

	Returns the exact same 17-tuple as your original function.
	"""

	audio_cache = AudioCache(audio_base_dir)

	audio_dict: Dict[int, int] = {}
	story_audio_goldstandard_sentences = defaultdict(list)
	story_audio_produced_sentences = defaultdict(list)

	rows: List[Dict[str, Any]] = []

	# Sentence-level collectors
	sentence_segments_path_list: List[str] = []
	sentence_segments_transcript_list: List[str] = []
	sentence_segments_intended_words_list: List[str] = []
	sentence_segments_ipa_transcript_list: List[str] = []
	sentence_segments_intended_words_list: List[str] = [] ## For later on analyzing word omission
	sentence_segments_goldstandard_list: List[str] = []
	sentence_segments_child_list: List[str] = []
	sentence_segments_grade_list: List[str] = []
	sentence_segments_error_category_list: List[str] = []
	sentence_segments_error_labels_list: List[str] = []

	# Re-constructing utterance window
	utterance_start_time_list = []
	utterance_end_time_list = []
	utterance_closestart_time_list = []
	utterance_closeend_time_list = []

	# Word-level collectors
	word_segments_path_list: List[str] = []
	word_segments_transcript_list: List[str] = []
	word_segments_ipa_transcript_list: List[str] = []
	word_segments_intended_words_list: List[str] = []
	word_segments_produced_words_list: List[str] = []
	word_segments_IPA_list: List[str] = []
	word_segments_error_category_list: List[str] = []
	word_segments_error_labels_list: List[List[str]] = []
	word_segments_child_list: List[str] = []
	word_segments_grade_list: List[str] = []

	# N-gram collectors
	word_segments_ngram_path_list: List[str] = []
	word_segments_ngram_transcript_list: List[str] = []
	word_segments_ngram_ipa_transcript_list: List[str] = []
	word_segments_ngram_child_list: List[str] = []
	word_segments_ngram_grade_list: List[str] = []

	# List of original audio names
	original_audio_name_list: List[str] = []

	# Compare close vs original time; this is because annotations of word-level tasks might exceed the timestamps of the sentence-level annotations.
	start_time_diff_list: List[float] = []
	end_time_diff_list: List[float] = []
	extreme_cases: List[List] = []

	for item in data:
		task_id = item.get("id")

		# debug only test 195798234
	#    if debug and task_id != 195798234:
	#        continue

		# cross-annotation count
		audio_dict[task_id] = audio_dict.get(task_id, 0) + 1

		annotations = item.get("annotations", [])
		if not annotations:
			continue

		goldStandard = item.get("data", {}).get("goldStandard")
		if task_id == 185849965:
			goldStandard = "Pat had a pan."
		if task_id == 195800625:
			goldStandard = "You won't escape me, Jake! Nate shouted, his voice a booming echo in the forest."

		original_audio_name = item.get("data", {}).get("original_audio_name")
		userID = original_audio_name.split('_')[1]
		storyID = original_audio_name.split('_')[3]
		grade = userID_grade_dict[storyID + ' ' + userID]
		story_audio_goldstandard_sentences[original_audio_name].append(goldStandard)

		sentence_level_id = item.get("data", {}).get("sentence_level_id")

		repeated_bool = item.get("data", {}).get("repeated")
		repeated = 'true' if repeated_bool else 'false'
		identifier = f"{original_audio_name} {goldStandard} {repeated}"

		# try to read pre-computed utterance window (kept to match original behavior)
		if task_id == 195800625:
			print(sentenceLabels_original_audio_timestamps[identifier])
		try:
			utterance_start_time, utterance_end_time = sentenceLabels_original_audio_timestamps[identifier]
		except Exception:
			utterance_start_time = None
			utterance_end_time = None

		# Load audio once
		audio_data, audio_duration_s = audio_cache.get(original_audio_name)

		# Export sentence clip if a pre-computed window exists
		utterance_output_filename = ''
		if (utterance_start_time is not None and
			utterance_end_time is not None and
			sentence_level_id is not None):
			utterance_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_sentence_segment.wav"
			export_clip(
				audio_data,
				utterance_start_time,
				utterance_end_time,
				os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename)
			)


		# Use the latest annotation block
		ann0 = annotations[0]
		grouped = parse_grouped_results(ann0, goldStandard, original_audio_name)
		# assign task_id onto grouped rows (saves passing again)
		for g in grouped.values():
			g["task_id"] = task_id

		# Build sorted per-word rows
		sorted_word_segments = sorted_word_rows_from_grouped(task_id, grouped)
		rows.extend(sorted_word_segments)

		# Collecting word-level segments and transcripts
		intended_words_list: List[str] = []
		produced_utterance: List[str] = []
		produced_ipa_sequence: List[str] = [] ## Full transcripts in IPA form
		error_category_sequence: List[str] = []
		error_labels_sequence: List[str] = []

		actual_buffered_audio_url = item.get("data", {}).get("audio", '')

		relative_start = max(0, utterance_start_time - word_offset)

		if actual_buffered_audio_url:

			start_time_list: List[float] = []
			end_time_list: List[float] = []

			for z, info in enumerate(sorted_word_segments, start=1):
				if info['start'] is None or info['end'] is None or sentence_level_id is None:
					continue

				start_time = info['start'] + relative_start
				end_time = info['end'] + relative_start
				start_time_list.append(start_time)
				end_time_list.append(end_time)
				if 'Parental Aid' not in info['error_labels']:
					intended_words_list.append(info['intended_words'].strip())
				produced_utterance.append(info['actual_production'].strip())
				produced_ipa_sequence.append(info['actual_ipa'])
				word_error_category = info['error_category']
				word_error_labels = info['error_labels']
				if '+' in word_error_category:
					word_error_category = word_error_category.split('+')
					modified_word_error_category = []
					for category in word_error_category:
						try:
							category = error_map[category]
						except:
							category = category
						modified_word_error_category.append(category)
					word_error_category = '+'.join(modified_word_error_category)
				error_category_sequence.append(word_error_category)
				if word_error_labels:
					error_labels_sequence.append('+'.join(word_error_labels))

				# Export per-word
				word_output_filename = f"{original_audio_name.split('.')[0]}_{task_id}_word_segment_{z}.wav"
				export_clip(
					audio_data,
					start_time,
					end_time,
					os.path.join(WORD_SEGMENTS_DIR, word_output_filename)
				)

				# Collect paths & metadata
				word_segments_path_list.append(os.path.join(WORD_SEGMENTS_DIR, word_output_filename))
				word_segments_transcript_list.append(info['actual_production'])
				word_segments_ipa_transcript_list.append(info['actual_ipa'])
				word_segments_intended_words_list.append(info['intended_words'])
				word_segments_produced_words_list.append(info['produced_word'])
				ipa_str = ''.join(info['IPA'][0]) if info['IPA'] else ''
				word_segments_IPA_list.append(ipa_str)
				word_segments_error_category_list.append(info['error_category'])
				word_segments_error_labels_list.append(info['error_labels'])
				word_segments_child_list.append(userID)
				word_segments_grade_list.append(grade)

				# N-gram window
				if NGRAM == 'full':
					ngram_start_time = start_time_list[0]
					ngram_end_time = end_time
					ngram_produced_utterance = ' '.join(produced_utterance)
					ngram_produced_ipa_sequence = ' '.join(produced_ipa_sequence)
				else:
					n = int(NGRAM)
					ngram_start_time = ''
					try:
						ngram_start_time = start_time_list[-n]
					except:
						ngram_start_time = start_time_list[0]
					ngram_end_time = end_time
					ngram_produced_utterance = ' '.join(produced_utterance[-n:])
					ngram_produced_ipa_sequence = ' '.join(produced_ipa_sequence[-n:])

				ngram_output_filename = f"{original_audio_name.split('.')[0]}_{task_id}_ngram_segment_{z}.wav"
				export_clip(
					audio_data,
					ngram_start_time,
					ngram_end_time,
					os.path.join(WORD_SEGMENTS_NGRAM_DIR, ngram_output_filename)
				)
				word_segments_ngram_path_list.append(os.path.join(WORD_SEGMENTS_NGRAM_DIR, ngram_output_filename))
				word_segments_ngram_transcript_list.append(ngram_produced_utterance)
				word_segments_ngram_ipa_transcript_list.append(ngram_produced_ipa_sequence)
				word_segments_ngram_child_list.append(userID)
				word_segments_ngram_grade_list.append(grade)

		# If no pre-computed utterance window, derive from word spans and export
		close_utterance_start_time = ''
		close_utterance_end_time = ''
		if sentence_level_id is not None:
			dstart, dend = derive_sentence_window_from_words(sorted_word_segments, audio_duration_s, buffer_s=0.010)
			if dstart is not None and dend is not None:
				close_utterance_start_time = dstart+relative_start
				close_utterance_end_time = dend+relative_start
				
				utterance_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_sentence_segment_close.wav"
				if debug and (close_utterance_start_time < utterance_start_time or close_utterance_end_time > utterance_end_time):
				#    print("Warning: the word segments extend past the original sentence audio")
				#    print('Original:', utterance_start_time, utterance_end_time)
				#    print("Close:", close_utterance_start_time, close_utterance_end_time)
				#    print(task_id, utterance_output_filename)
				#    print('\n')
					start_time_diff = abs(close_utterance_start_time - utterance_start_time)
					end_time_diff = abs(close_utterance_end_time - utterance_end_time)
					start_time_diff_list.append(start_time_diff)
					end_time_diff_list.append(end_time_diff)

					if start_time_diff >= 1 or end_time_diff >= 1:
						extreme_cases.append([task_id, round(start_time_diff, 3), round(end_time_diff, 3)])

				export_clip(
					audio_data,
					close_utterance_start_time,
					close_utterance_end_time,
					os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename)
				)

				original_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_sentence_segment.wav"
				if original_output_filename in os.listdir(SENTENCE_SEGMENTS_DIR):
					os.system('rm ' + SENTENCE_SEGMENTS_DIR + '/' + original_output_filename)

		# Sentence-level collection
		if produced_utterance:
			sentence_segments_path_list.append(os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename))
			produced_sentence = ' '.join(produced_utterance)
			sentence_segments_transcript_list.append(produced_sentence)
			intended_words_sequence = ' '.join(intended_words_list)
			sentence_segments_intended_words_list.append(intended_words_sequence)
			produced_sentence_ipa = ' '.join(produced_ipa_sequence)
			sentence_segments_ipa_transcript_list.append(produced_sentence_ipa)
			sentence_segments_goldstandard_list.append(goldStandard)
			sentence_segments_child_list.append(userID)
			sentence_segments_grade_list.append(grade)
			sentence_segments_error_category_list.append(error_category_sequence)
			sentence_segments_error_labels_list.append(error_labels_sequence)
			original_audio_name_list.append(original_audio_name)
			utterance_start_time_list.append(utterance_start_time)
			utterance_end_time_list.append(utterance_end_time)
			utterance_closestart_time_list.append(close_utterance_start_time)
			utterance_closeend_time_list.append(close_utterance_end_time)

			story_audio_produced_sentences[original_audio_name].append(produced_sentence)

		# Assertions
		assert len(sentence_segments_transcript_list) == len(sentence_segments_path_list), "Mismatch in sentence segments and transcripts length"
		assert len(word_segments_path_list) == len(word_segments_transcript_list), "Mismatch in word segments and transcripts length"
		assert len(word_segments_path_list) == len(word_segments_error_category_list), "Mismatch in word segments and error categories length"
		assert len(word_segments_path_list) == len(word_segments_error_labels_list), "Mismatch in word segments and error labels length"
		assert len(word_segments_ngram_path_list) == len(word_segments_ngram_transcript_list), "Mismatch in n-gram word segments and transcripts length"
		assert len(word_segments_ngram_path_list) == len(word_segments_error_category_list), "Mismatch in n-gram word segments and word segments length"

	print(f'Start time difference: mean {statistics.mean(start_time_diff_list)} std: {statistics.stdev(start_time_diff_list)}')
	print(f'End time difference: mean {statistics.mean(end_time_diff_list)} std: {statistics.stdev(end_time_diff_list)}')
	print('\n')
	print('EXTREME CASES')
	for case in extreme_cases:
		print(case)

	return (
		audio_dict,
		story_audio_goldstandard_sentences,
		story_audio_produced_sentences,
		rows,
		sentence_segments_path_list,
		sentence_segments_transcript_list,
		sentence_segments_intended_words_list,
		sentence_segments_ipa_transcript_list,
		sentence_segments_goldstandard_list,
		sentence_segments_child_list,
		sentence_segments_grade_list,
		sentence_segments_error_category_list,
		sentence_segments_error_labels_list,
		word_segments_path_list,
		word_segments_transcript_list,
		word_segments_ipa_transcript_list,
		word_segments_intended_words_list,
		word_segments_produced_words_list,
		word_segments_IPA_list,
		word_segments_error_category_list,
		word_segments_error_labels_list,
		word_segments_child_list,
		word_segments_grade_list,
		word_segments_ngram_path_list,
		word_segments_ngram_transcript_list,
		word_segments_ngram_ipa_transcript_list,
		word_segments_ngram_child_list,
		word_segments_ngram_grade_list,
		utterance_start_time_list,
		utterance_end_time_list,
		utterance_closestart_time_list,
		utterance_closeend_time_list,
		original_audio_name_list,
		error_dict,
	)

		
