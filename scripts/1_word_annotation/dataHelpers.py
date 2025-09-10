from collections import defaultdict
import os
from pydub import AudioSegment
import soundfile as sf

def build_segments_and_rows(
    data,
    audio_base_dir,
    SENTENCE_SEGMENTS_DIR,
    WORD_SEGMENTS_DIR,
    WORD_SEGMENTS_NGRAM_DIR,
    sentenceLabels_original_audio_timestamps,
    NGRAM,
    error_dict,
    debug=False
):
    """
    Processes Label Studio JSON 'data' into sentence/word audio segments and metadata rows.

    Parameters
    ----------
    data : list[dict]
    audio_base_dir : str
    SENTENCE_SEGMENTS_DIR : str
    WORD_SEGMENTS_DIR : str
    WORD_SEGMENTS_NGRAM_DIR : str
    sentenceLabels_original_audio_timestamps : dict
        { "<original_audio> <goldStandard> <repeated>": [start_time, end_time] }
    NGRAM : 'full' | int
    error_dict : dict[str, list]
        Collected issues per annotator (will be appended to).

    Returns
    -------
    tuple
        (audio_dict,
         story_audio_goldstandard_sentences,
         story_audio_produced_sentences,
         rows,
         sentence_segments_path_list,
         sentence_segments_transcript_list,
         sentence_segments_goldstandard_list,
         word_segments_path_list,
         word_segments_transcript_list,
         word_segments_intended_words_list,
         word_segments_produced_words_list,
         word_segments_IPA_list,
         word_segments_error_category_list,
         word_segments_error_labels_list,
         word_segments_ngram_path_list,
         word_segments_ngram_transcript_list,
         error_dict)
    """

    audio_dict = {}  # For comparing cross-annotations later
    story_audio_goldstandard_sentences = defaultdict(list)
    story_audio_produced_sentences = defaultdict(list)

    rows = []

    # Sentence-level data collectors
    sentence_segments_path_list = []
    sentence_segments_transcript_list = []
    sentence_segments_goldstandard_list = []

    # Word-level data collectors
    word_segments_path_list = []
    word_segments_transcript_list = []
    word_segments_intended_words_list = []
    word_segments_produced_words_list = []
    word_segments_IPA_list = []
    word_segments_error_category_list = []
    word_segments_error_labels_list = []

    # N-gram word-level collectors
    word_segments_ngram_path_list = []
    word_segments_ngram_transcript_list = []

    for item in data:
        task_id = item.get("id")

        #debug thing
        if debug and task_id != 195798234:
            continue

        # Track how many times a task appears
        if task_id not in audio_dict:
            audio_dict[task_id] = 1
        else:
            audio_dict[task_id] += 1

        annotations = item.get("annotations", [])
        if not annotations:
            continue

        goldStandard = item.get("data", {}).get("goldStandard")
        if task_id == 185849965:
            goldStandard = "Pat had a pan."

        original_audio_name = item.get("data", {}).get("original_audio_name")

        if original_audio_name not in story_audio_goldstandard_sentences:
            story_audio_goldstandard_sentences[original_audio_name] = [goldStandard]
        else:
            story_audio_goldstandard_sentences[original_audio_name].append(goldStandard)

        sentence_level_id = item.get("data", {}).get("sentence_level_id")

        repeated_bool = item.get("data", {}).get("repeated")
        repeated = 'true' if repeated_bool else 'false'
        identifier = original_audio_name + ' ' + goldStandard + ' ' + repeated
        utterance_start_time = None
        utterance_end_time = None
        try:
            utterance_timestamps = sentenceLabels_original_audio_timestamps[identifier]
            utterance_start_time = utterance_timestamps[0]
            utterance_end_time = utterance_timestamps[1]
        except Exception:
            # Keep as-is to match original behavior
            pass

        # Outputting sentence-level segments
        audio_file = os.path.join(audio_base_dir, original_audio_name)
        audio_data = AudioSegment.from_file(audio_file)

        utterance_output_filename = ''
        if utterance_start_time is not None and utterance_end_time is not None and sentence_level_id is not None:
            sentence_segment = audio_data[utterance_start_time * 1000 : utterance_end_time * 1000]
            utterance_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_sentence_segment.wav"
            sentence_segment.export(os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename), format='wav')

        # Use the latest annotation block
        result = annotations[0].get("result", [])
        annotator = annotations[0].get("completed_by", {}).get("email")

        # Group all parts of a word using result.id
        grouped = defaultdict(lambda: {
            "task_id": task_id,
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
            "actual_production": ''
        })

        if result != []:
            for r in result:
                result_id = r.get("id")
                value = r.get("value", {})
                r_type = r.get("type")
                from_name = r.get("from_name")

                if r_type == "labels" and from_name == "WordAnnotation":
                    grouped[result_id]["start"] = value.get("start")
                    grouped[result_id]["end"] = value.get("end")
                    category = value.get("labels")[0]
                    grouped[result_id]["error_category"] = category
                    grouped[result_id]["original_error_category"] = category

                if grouped[result_id]["original_error_category"] != "Mixed Error" and r_type == "choices":
                    category = grouped[result_id]["error_category"]
                    if category != '' and category != "Mixed Error":
                        category_no_space = category.replace(' ', '')
                        if category_no_space in from_name:
                            labels = value.get("choices", [])
                            grouped[result_id]["error_labels"].extend(labels)

                if r_type == "textarea" and from_name == "spoken_words":
                    category = grouped[result_id]["original_error_category"]
                    if category in ["Run-on", "Contraction/Shortening"]:
                        grouped[result_id]["intended_words"].extend(value.get("text", []))

                if r_type == "textarea" and from_name == "spoken_word":
                    category = grouped[result_id]["original_error_category"]
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
                        grouped[result_id]["intended_words"].extend(value.get("text", []))

                if r_type == "textarea" and from_name == "produced_word":
                    category = grouped[result_id]["original_error_category"]
                    if category in [
                        "Contraction/Shortening",
                        "Visual Tracking Error",
                        "Self Response",
                        "Other",
                        "Orthographic Error",
                        "Grammatical Error",
                        "Structural Error",
                        "Mixed Error"
                    ]:
                        grouped[result_id]["produced_word"].extend(value.get("text", []))

                if r_type == "textarea" and from_name == "mispronunciation_word":
                    category = grouped[result_id]["original_error_category"]
                    if category in [
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
                        grouped[result_id]["IPA"].extend(value.get("text", []))

                if r_type == "textarea" and from_name == "issues":
                    grouped[result_id]["comments"].extend(value.get("text", []))

                if grouped[result_id]["original_error_category"] == "Mixed Error" and r_type == "taxonomy":
                    category = grouped[result_id]["original_error_category"]
                    if category == "Mixed Error":
                        labels = []
                        mixed = value.get("taxonomy", [])
                        for tok in mixed:
                            if tok[0] not in category:
                                category = category + '+' + tok[0]
                            if len(tok) > 1:
                                labels = labels + tok[1:]
                        grouped[result_id]["error_category"] = category
                        grouped[result_id]["error_labels"] = labels

        # Create a row for each word
        word_segments = []
        for word_id, info in grouped.items():
            if (
                utterance_output_filename != '' and
                info['error_labels'] != ["Unfilled Pause"] and
                info['start'] is not None and
                info['end'] is not None and
                sentence_level_id is not None
            ):
                # Cleaning IPA
                if info["IPA"] != []:
                    while '/' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('/', '')
                    while 'd͡ʒ' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('d͡ʒ', 'dʒ')
                    while 't͡ʃ' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('t͡ʃ', 'tʃ')

                actual_production = ''
                error_category = info['error_category']
                intended_words = ' '.join(info["intended_words"])
                produced_word = ' '.join(info["produced_word"])
                IPA = info["IPA"]

                if error_category == 'Correct':
                    actual_production = intended_words
                else:
                    if intended_words != '' and produced_word == '' and IPA == []:
                        actual_production = intended_words
                    elif intended_words != '' and produced_word == '' and IPA != []:
                        actual_production = IPA[0]
                    elif intended_words != '' and produced_word != '' and IPA == []:
                        actual_production = produced_word
                    elif intended_words != '' and produced_word != '' and IPA != []:
                        actual_production = produced_word
                    elif intended_words == '' and produced_word == '' and IPA != []:
                        actual_production = IPA[0]
                    elif intended_words == '' and produced_word != '' and IPA == []:
                        actual_production = produced_word
                    elif intended_words == '' and produced_word != '' and IPA != []:
                        actual_production = produced_word
                    else:
                        pass

                word_segments.append({
                    "task_id": task_id,
                    "annotator": annotator,
                    "start": info["start"],
                    "end": info["end"],
                    "intended_words": ' '.join(info["intended_words"]),
                    "produced_word": ' '.join(info["produced_word"]),
                    "IPA": info["IPA"],
                    "error_category": info["error_category"],
                    "error_labels": info["error_labels"],
                    "comments": " | ".join(info["comments"]),
                    "goldStandard": goldStandard,
                    "original_audio_name": original_audio_name,
                    "actual_production": actual_production
                })

        # Sort by start time
        sorted_word_segments = sorted(word_segments, key=lambda x: (x["start"] is None, x["start"]))
        rows.extend(sorted_word_segments)

        # Collecting sentence-level segments and transcripts
        produced_utterance = []

        # Collecting word-level segments and transcripts
        if utterance_output_filename != '':
            sentence_audio_file = os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename)
            sentence_audio_data, sample_rate = sf.read(sentence_audio_file)
            start_time_list = []
            end_time_list = []
            first_word_start_time = 0
            for z in range(len(sorted_word_segments)):
                info = sorted_word_segments[z]
                if info['start'] is not None and info['end'] is not None and sentence_level_id is not None:
                    if z == 0:
                        first_word_start_time = info['start']
                    start_time = info['start'] - first_word_start_time
                    end_time = info['end'] - first_word_start_time
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    actual_production = info['actual_production']
                    produced_utterance.append(actual_production)

                    # Convert to samples
                    start_sample = int(round(start_time * sample_rate))
                    end_sample   = int(round(end_time * sample_rate))

                    # Slice directly by samples
                    word_segment = (
                        sentence_audio_data[start_sample:end_sample, :]
                        if sentence_audio_data.ndim > 1
                        else sentence_audio_data[start_sample:end_sample]
                    )
                    word_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_word_segment_{z+1}.wav"
                    sf.write(os.path.join(WORD_SEGMENTS_DIR, word_output_filename), word_segment, sample_rate)

                    # DEBUG file (kept as-is)
                    sf.write('test.wav', sentence_audio_data[0: int(round(1.1024390243902435*sample_rate))], sample_rate)

                    word_segments_path_list.append(os.path.join(WORD_SEGMENTS_DIR, word_output_filename))
                    word_segments_transcript_list.append(actual_production)
                    word_segments_intended_words_list.append(info['intended_words'])
                    word_segments_produced_words_list.append(info['produced_word'])
                    try:
                        word_segments_IPA_list.append(''.join(info['IPA'][0]))
                    except Exception:
                        word_segments_IPA_list.append('')
                    word_segments_error_category_list.append(info['error_category'])
                    word_segments_error_labels_list.append(info['error_labels'])

                    ## Collecting n-gram word-level segments
                    if NGRAM == 'full':
                        ngram_start_time = start_time_list[0]
                        ngram_end_time = end_time
                        ngram_produced_utterance = ' '.join(produced_utterance)
                    else:
                        n = int(NGRAM)
                        ngram_start_time = start_time_list[-1 * n]
                        ngram_end_time = end_time
                        ngram_produced_utterance = ' '.join(produced_utterance[-1 * n : ])

                    ngram_start_sample = int(round(ngram_start_time * sample_rate))
                    ngram_end_sample = int(round(ngram_end_time * sample_rate))
                    ngram_segment = (
                        sentence_audio_data[ngram_start_sample:ngram_end_sample, :]
                        if sentence_audio_data.ndim > 1
                        else sentence_audio_data[ngram_start_sample:ngram_end_sample]
                    )
                    ngram_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_ngram_segment_{z+1}.wav"
                    sf.write(os.path.join(WORD_SEGMENTS_NGRAM_DIR, ngram_output_filename), ngram_segment, sample_rate)
                    word_segments_ngram_path_list.append(os.path.join(WORD_SEGMENTS_NGRAM_DIR, ngram_output_filename))
                    word_segments_ngram_transcript_list.append(ngram_produced_utterance)

        if produced_utterance != []:
            sentence_segments_path_list.append(os.path.join(SENTENCE_SEGMENTS_DIR, utterance_output_filename))
            sentence_segments_transcript_list.append(' '.join(produced_utterance))
            sentence_segments_goldstandard_list.append(goldStandard)
            if original_audio_name not in story_audio_produced_sentences:
                story_audio_produced_sentences[original_audio_name] = [' '.join(produced_utterance)]
            else:
                story_audio_produced_sentences[original_audio_name].append(' '.join(produced_utterance))

        # Assertions (kept as in your original)
        assert len(sentence_segments_transcript_list) == len(sentence_segments_path_list), "Mismatch in sentence segments and transcripts length"
        assert len(word_segments_path_list) == len(word_segments_transcript_list), "Mismatch in word segments and transcripts length"
        assert len(word_segments_path_list) == len(word_segments_error_category_list), "Mismatch in word segments and error categories length"
        assert len(word_segments_path_list) == len(word_segments_error_labels_list), "Mismatch in word segments and error labels length"
        assert len(word_segments_ngram_path_list) == len(word_segments_ngram_transcript_list), "Mismatch in n-gram word segments and transcripts length"
        assert len(word_segments_ngram_path_list) == len(word_segments_error_category_list), "Mismatch in n-gram word segments and word segments length"

    return (
        audio_dict,
        story_audio_goldstandard_sentences,
        story_audio_produced_sentences,
        rows,
        sentence_segments_path_list,
        sentence_segments_transcript_list,
        sentence_segments_goldstandard_list,
        word_segments_path_list,
        word_segments_transcript_list,
        word_segments_intended_words_list,
        word_segments_produced_words_list,
        word_segments_IPA_list,
        word_segments_error_category_list,
        word_segments_error_labels_list,
        word_segments_ngram_path_list,
        word_segments_ngram_transcript_list,
        error_dict,
    )
