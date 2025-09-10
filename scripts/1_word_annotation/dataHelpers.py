import os
from collections import defaultdict
from pydub import AudioSegment
import soundfile as sf

def process_tasks(
    data,
    sentence_timestamps,
    audio_base_dir,
    sentence_segments_dir,
    word_segments_dir,
    word_segments_ngram_dir,
    ngram="full",
    error_dict=None,
):
    """
    Process Label Studio JSON tasks to:
      - export sentence-level WAV segments
      - export word-level and n-gram WAV segments
      - collect transcripts & annotations
      - run overlap checks
      - accumulate error messages per annotator

    Parameters
    ----------
    data : list[dict]
        Parsed Label Studio JSON export.
    sentence_timestamps : dict
        Map: "<original_audio_name> <goldStandard> <repeated>" -> [start_time, end_time].
    audio_base_dir : str
        Folder containing the original story/session audio files.
    sentence_segments_dir : str
        Output folder for sentence-level audio segments.
    word_segments_dir : str
        Output folder for word-level audio segments.
    word_segments_ngram_dir : str
        Output folder for n-gram audio segments.
    ngram : "full" | int
        If "full", each n-gram grows from first word; if int, fixed-length window.
    error_dict : dict or defaultdict(list) | None
        If provided, errors will be appended here; if None, a defaultdict(list) is used.

    Returns
    -------
    dict
        {
          "rows": list[dict],
          "story_audio_goldstandard_sentences": dict[str, list[str]],
          "story_audio_produced_sentences": dict[str, list[str]],
          "sentence_segments": {"paths": [], "transcripts": [], "gold": []},
          "word_segments": {
              "paths": [], "transcripts": [], "intended": [], "produced": [],
              "ipa": [], "category": [], "labels": []
          },
          "ngram_segments": {"paths": [], "transcripts": []},
          "audio_task_counts": dict[task_id -> count],
          "errors": dict[email -> list]
        }
    """
    if error_dict is None:
        error_dict = defaultdict(list)

    # collectors
    story_audio_goldstandard_sentences = defaultdict(list)
    story_audio_produced_sentences = defaultdict(list)

    sentence_paths, sentence_transcripts, sentence_gold = [], [], []

    word_paths, word_transcripts = [], []
    word_intended, word_produced, word_ipa = [], [], []
    word_category, word_labels = [], []

    ngram_paths, ngram_transcripts = [], []

    rows = []
    audio_task_counts = {}

    for item in data:
        task_id = item.get("id")

        # how many times this task appears (for cross-annotation/bookkeeping)
        audio_task_counts[task_id] = audio_task_counts.get(task_id, 0) + 1

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
        repeated = item.get("data", {}).get("repeated")
        repeated = 'true' if repeated else 'false'

        identifier = f"{original_audio_name} {goldStandard} {repeated}"
        utterance_start_time = None
        utterance_end_time = None
        try:
            utterance_start_time, utterance_end_time = sentence_timestamps[identifier]
        except Exception:
            # special cases handled upstream; skip if not found
            pass

        # ---- sentence-level audio slice
        utterance_output_filename = ''
        if utterance_start_time is not None and utterance_end_time is not None and sentence_level_id is not None:
            audio_file = os.path.join(audio_base_dir, original_audio_name)
            audio_data = AudioSegment.from_file(audio_file)
            sentence_segment = audio_data[utterance_start_time * 1000 : utterance_end_time * 1000]
            utterance_output_filename = f"{os.path.splitext(original_audio_name)[0]}_{task_id}_sentence_segment.wav"
            sentence_segment.export(os.path.join(sentence_segments_dir, utterance_output_filename), format='wav')

        # Only the first (latest) annotation block
        result = annotations[0].get("result", [])
        annotator = annotations[0].get("completed_by", {}).get("email")

        # ---- group per word-id
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

        if result:
            for r in result:
                rid = r.get("id")
                val = r.get("value", {})
                r_type = r.get("type")
                from_name = r.get("from_name")

                if r_type == "labels" and from_name == "WordAnnotation":
                    grouped[rid]["start"] = val.get("start")
                    grouped[rid]["end"] = val.get("end")
                    category = val.get("labels", [None])[0]
                    grouped[rid]["error_category"] = category
                    grouped[rid]["original_error_category"] = category

                if grouped[rid]["original_error_category"] != "Mixed Error" and r_type == "choices":
                    category = grouped[rid]["error_category"]
                    if category and category != "Mixed Error":
                        if category.replace(' ', '') in (from_name or ""):
                            labels = val.get("choices", [])
                            grouped[rid]["error_labels"].extend(labels)

                if r_type == "textarea" and from_name == "spoken_words":
                    if grouped[rid]["original_error_category"] in ["Run-on", "Contraction/Shortening"]:
                        grouped[rid]["intended_words"].extend(val.get("text", []))

                if r_type == "textarea" and from_name == "spoken_word":
                    if grouped[rid]["original_error_category"] in [
                        "Correct", "Phonological Error", "Orthographic Error",
                        "Grammatical Error", "Structural Error", "Visual Tracking Error",
                        "Disfluency Error", "Mixed Error",
                    ] or "Mixed Error+" in (grouped[rid]["original_error_category"] or ""):
                        grouped[rid]["intended_words"].extend(val.get("text", []))

                if r_type == "textarea" and from_name == "produced_word":
                    if grouped[rid]["original_error_category"] in [
                        "Contraction/Shortening", "Visual Tracking Error", "Self Response",
                        "Other", "Orthographic Error", "Grammatical Error",
                        "Structural Error", "Mixed Error"
                    ]:
                        grouped[rid]["produced_word"].extend(val.get("text", []))

                if r_type == "textarea" and from_name == "mispronunciation_word":
                    if grouped[rid]["original_error_category"] in [
                        "Contraction/Shortening", "Other", "Phonological Error",
                        "Orthographic Error", "Grammatical Error", "Structural Error",
                        "Visual Tracking Error", "Disfluency Error", "Mixed Error"
                    ]:
                        grouped[rid]["IPA"].extend(val.get("text", []))

                if r_type == "textarea" and from_name == "issues":
                    grouped[rid]["comments"].extend(val.get("text", []))

                if grouped[rid]["original_error_category"] == "Mixed Error" and r_type == "taxonomy":
                    category = "Mixed Error"
                    labels = []
                    for tok in val.get("taxonomy", []):
                        if tok[0] not in category:
                            category += '+' + tok[0]
                        if len(tok) > 1:
                            labels += tok[1:]
                    grouped[rid]["error_category"] = category
                    grouped[rid]["error_labels"] = labels

        # ---- build word rows
        word_rows = []
        for wid, info in grouped.items():
            if (
                utterance_output_filename != ''
                and info['error_labels'] != ["Unfilled Pause"]
                and info['start'] is not None
                and info['end'] is not None
                and sentence_level_id is not None
            ):
                # clean IPA
                if info["IPA"]:
                    while '/' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('/', '')
                    while 'd͡ʒ' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('d͡ʒ', 'dʒ')
                    while 't͡ʃ' in info["IPA"][0]:
                        info["IPA"][0] = info["IPA"][0].replace('t͡ʃ', 'tʃ')

                error_category = info['error_category']
                intended_words = ' '.join(info["intended_words"])
                produced_word = ' '.join(info["produced_word"])
                IPA = info["IPA"]

                actual_production = ''
                if error_category == 'Correct':
                    actual_production = intended_words
                else:
                    if intended_words and not produced_word and not IPA:
                        actual_production = intended_words
                    elif intended_words and not produced_word and IPA:
                        actual_production = IPA[0]
                    elif intended_words and produced_word:
                        actual_production = produced_word
                    elif not intended_words and produced_word:
                        actual_production = produced_word
                    elif not intended_words and not produced_word and IPA:
                        actual_production = IPA[0]

                word_rows.append({
                    "task_id": task_id,
                    "annotator": annotator,
                    "start": info["start"],
                    "end": info["end"],
                    "intended_words": intended_words,
                    "produced_word": produced_word,
                    "IPA": info["IPA"],
                    "error_category": info["error_category"],
                    "error_labels": info["error_labels"],
                    "comments": " | ".join(info["comments"]),
                    "goldStandard": goldStandard,
                    "original_audio_name": original_audio_name,
                    "actual_production": actual_production
                })

        # sort by start and extend master list
        sorted_word_segments = sorted(word_rows, key=lambda x: (x["start"] is None, x["start"]))
        rows.extend(sorted_word_segments)

        # ---- audio slicing for words / n-grams and sentence transcript
        produced_utterance = []
        if utterance_output_filename != '':
            sentence_audio_path = os.path.join(sentence_segments_dir, utterance_output_filename)
            sentence_audio_data, sample_rate = sf.read(sentence_audio_path)

            start_time_list, end_time_list = [], []
            first_word_start_time = 0

            for z, info in enumerate(sorted_word_segments):
                if info['start'] is not None and info['end'] is not None and sentence_level_id is not None:
                    if z == 0:
                        first_word_start_time = info['start']
                    start_time = info['start'] - first_word_start_time
                    end_time = info['end'] - first_word_start_time
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)

                    actual_production = info['actual_production']
                    produced_utterance.append(actual_production)

                    # slice word segment
                    start_sample = int(round(start_time * sample_rate))
                    end_sample = int(round(end_time * sample_rate))
                    word_segment = (
                        sentence_audio_data[start_sample:end_sample, :]
                        if sentence_audio_data.ndim > 1
                        else sentence_audio_data[start_sample:end_sample]
                    )
                    word_output_filename = f"{os.path.splitext(original_audio_name)[0]}_{task_id}_word_segment_{z+1}.wav"
                    word_output_path = os.path.join(word_segments_dir, word_output_filename)
                    sf.write(word_output_path, word_segment, sample_rate)

                    # metadata
                    word_paths.append(word_output_path)
                    word_transcripts.append(actual_production)
                    word_intended.append(info['intended_words'])
                    word_produced.append(info['produced_word'])
                    try:
                        word_ipa.append(''.join(info['IPA'][0]))
                    except Exception:
                        word_ipa.append('')
                    word_category.append(info['error_category'])
                    word_labels.append(info['error_labels'])

                    # n-gram segment
                    if ngram == 'full':
                        ng_start = start_time_list[0]
                        ng_end = end_time
                        ng_trans = ' '.join(produced_utterance)
                    else:
                        n = int(ngram)
                        ng_start = start_time_list[-n]
                        ng_end = end_time
                        ng_trans = ' '.join(produced_utterance[-n:])

                    ng_start_sample = int(round(ng_start * sample_rate))
                    ng_end_sample = int(round(ng_end * sample_rate))
                    ngram_segment = (
                        sentence_audio_data[ng_start_sample:ng_end_sample, :]
                        if sentence_audio_data.ndim > 1
                        else sentence_audio_data[ng_start_sample:ng_end_sample]
                    )
                    ngram_output_filename = f"{os.path.splitext(original_audio_name)[0]}_{task_id}_ngram_segment_{z+1}.wav"
                    ngram_output_path = os.path.join(word_segments_ngram_dir, ngram_output_filename)
                    sf.write(ngram_output_path, ngram_segment, sample_rate)

                    ngram_paths.append(ngram_output_path)
                    ngram_transcripts.append(ng_trans)

        # sentence-level transcript row
        if produced_utterance:
            sentence_paths.append(os.path.join(sentence_segments_dir, utterance_output_filename))
            sentence_transcripts.append(' '.join(produced_utterance))
            sentence_gold.append(goldStandard)
            story_audio_produced_sentences[original_audio_name].append(' '.join(produced_utterance))

        # sanity checks (mirror your asserts)
        assert len(sentence_transcripts) == len(sentence_paths), "Mismatch in sentence segments and transcripts length"
        assert len(word_paths) == len(word_transcripts), "Mismatch in word segments and transcripts length"
        assert len(word_paths) == len(word_category), "Mismatch in word segments and error categories length"
        assert len(word_paths) == len(word_labels), "Mismatch in word segments and error labels length"
        assert len(ngram_paths) == len(ngram_transcripts), "Mismatch in n-gram word segments and transcripts length"
        assert len(ngram_paths) == len(word_category), "Mismatch in n-gram word segments and word segments length"

        # overlap check
        for i in range(len(sorted_word_segments) - 1):
            annot = sorted_word_segments[i]["annotator"]
            cur_info = sorted_word_segments[i]
            nxt_info = sorted_word_segments[i + 1]
            if cur_info["end"] is not None and nxt_info["start"] is not None:
                if cur_info["end"] > nxt_info["start"]:
                    error_dict[annot].append([task_id, annot, "Overlap detected between the following", cur_info, nxt_info, '\n'])

    # package outputs
    return {
        "rows": rows,
        "story_audio_goldstandard_sentences": dict(story_audio_goldstandard_sentences),
        "story_audio_produced_sentences": dict(story_audio_produced_sentences),
        "sentence_segments": {
            "paths": sentence_paths,
            "transcripts": sentence_transcripts,
            "gold": sentence_gold,
        },
        "word_segments": {
            "paths": word_paths,
            "transcripts": word_transcripts,
            "intended": word_intended,
            "produced": word_produced,
            "ipa": word_ipa,
            "category": word_category,
            "labels": word_labels,
        },
        "ngram_segments": {
            "paths": ngram_paths,
            "transcripts": ngram_transcripts,
        },
        "audio_task_counts": audio_task_counts,
        "errors": error_dict,
    }
