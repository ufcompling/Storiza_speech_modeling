#!/usr/bin/env python
# coding: utf-8

# **Date**: June 28, 2025
# 
# **Authors**: Zoey Liu, 陳宇賀
# 
# **Purpose** (1) automatic checking of some, but not all, annotation errors for Storiza project; (2) Output annotated story-level, sentence-level, and word-level segments and transcripts

# **Word-level JSON file checks and processing**

import os
import json
import csv
import string
from collections import defaultdict
import pandas as pd
from pydub import AudioSegment
import soundfile as sf


from directoryConfig import  *
from annotatorInformation import *
from jsonHelpers import *

# ---- Processing parameters ----
SAMPLE_RATE = 16000
NGRAM = "full"  # or an int (e.g., 3) if you want fixed-length n-grams
DEBUG = False #used to print additional data



# ----------- Load JSON ----------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

#============= (1) Examining JSON ================
if DEBUG:
    # ---------- Print First Example ----------
    print("🔍 Structure with example values from first item:\n")
    #print_structure_with_values(data[0])
    for item in data:
      task_id = item.get("id")
      if task_id == 195798234:
        print_structure_with_values(item)

    # ---------- Look At Annotation Types ----------
    types, from_names, pairs = get_annotation_type_sets(data)
    print(types)
    print(from_names)
    print(pairs)


#============= (2) Generate Data DICTs================
sentenceLabels_original_audio_timestamps = load_sentence_label_timestamps(SENTENCE_LABELS_CSV)

#print out some examples:
for item in sentenceLabels_original_audio_timestamps[:min(5, len(sentenceLabels_original_audio_timestamps))]:
    print_structure_with_values(item)

#try specific example:
try:
    print(sentenceLabels_original_audio_timestamps[""])
except KeyError:
    print("item not found")


# In[23]:


# Collecting annotated intended/goldstandard sentences for each audio
story_audio_goldstandard_sentences = defaultdict(list)
story_audio_produced_sentences = defaultdict(list)

# Story-level data
story_data = pd.DataFrame(columns=['Path', 'Transcript', 'goldStandard'])
story_data_path_list = []
story_data_transcript_list = []
story_data_goldstandard_list = []

# Sentence-level data
sentence_segments_data = pd.DataFrame(columns=['Path', 'Transcript', 'goldStandard']) 
sentence_segments_path_list = []
sentence_segments_transcript_list = []
sentence_segments_goldstandard_list = []
sentence_segments_path = '/Users/liu.ying/University of Florida/Leite,Walter - Storiza Corpus Spring 2025/Storiza_Participant_Recording_05-30-25/annotated_sentence_segments/'
try:
    os.makedirs(sentence_segments_path, exist_ok=True)
    print(f"Directory '{sentence_segments_path}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{sentence_segments_path}': {e}")

# Unigram word-level data
word_segments_data = pd.DataFrame(columns=['Path', 'Transcript', 'Intended Words', 'Produced Words', 'IPA', 'Error Category', 'Error Labels'])
word_segments_path_list = []
word_segments_transcript_list = []
word_segments_intended_words_list = []
word_segments_produced_words_list = []
word_segments_IPA_list = []
word_segments_error_category_list = []
word_segments_error_labels_list = []
word_segments_path = '/Users/liu.ying/University of Florida/Leite,Walter - Storiza Corpus Spring 2025/Storiza_Participant_Recording_05-30-25/annotated_word_segments/'
try:
    os.makedirs(word_segments_path, exist_ok=True)
    print(f"Directory '{word_segments_path}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{word_segments_path}': {e}")

# N-gram word-level data, taking context into account when doing classification later
NGRAM = 'full'
word_segments_data_ngram = pd.DataFrame(columns=['Path', 'Transcript', 'Intended Words', 'Produced Words', 'IPA', 'Error Category', 'Error Labels'])
word_segments_ngram_path_list = []
word_segments_ngram_transcript_list = []
word_segments_ngram_path = '/Users/liu.ying/University of Florida/Leite,Walter - Storiza Corpus Spring 2025/Storiza_Participant_Recording_05-30-25/annotated_word_segments_ngram/'
try:
    os.makedirs(word_segments_ngram_path, exist_ok=True)
    print(f"Directory '{word_segments_ngram_path}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{word_segments_ngram_path}': {e}")



error_dict = {}
for annotator in annotator_list:
  error_dict[annotator] = []

# -------- Extract & Format Data --------
rows = []
audio_dict = {} ## For comparing cross-annotations later

for item in data:
    task_id = item.get("id")

    ## Checking to see which task has more than one annotation
    audio = item.get("data").get("audio")
    if task_id not in audio_dict:
      audio_dict[task_id] = 1
    else:
      audio_dict[task_id] += 1

    annotations = item.get("annotations", [])
    if not annotations:
        continue

    goldStandard = item.get("data").get("goldStandard")
    if task_id == 185849965:
      goldStandard = "Pat had a pan."

    original_audio_name = item.get("data").get("original_audio_name")
    if original_audio_name not in story_audio_goldstandard_sentences:
      story_audio_goldstandard_sentences[original_audio_name] = [goldStandard]
    else:
      story_audio_goldstandard_sentences[original_audio_name].append(goldStandard)

    sentence_level_id = item.get("data").get("sentence_level_id")
    sample_rate = 16000

    repeated = item.get("data").get("repeated")
    if repeated:
      repeated = 'true'
    else:
      repeated = 'false'
    identifier = original_audio_name + ' ' + goldStandard + ' ' + repeated
    utterance_start_time = None
    utterance_end_time = None
    try:
      utterance_timestamps = sentenceLabels_original_audio_timestamps[identifier]
      utterance_start_time = utterance_timestamps[0]
      utterance_end_time = utterance_timestamps[1]
    except:
      pass # At the end, remember to manually add a space before Want for this sentence: "Hi, Dawn!" hooted Dale,"Want to leap and play?"

    ## Outputting sentence-level segments
    audio_file = os.path.join(audio_base_dir, original_audio_name)
#    audio_data, sample_rate = sf.read(audio_file)  # data = numpy array
    audio_data = AudioSegment.from_file(audio_file)
#    utterance_start_time = item.get("data").get("start_time")
#    utterance_end_time = item.get("data").get("end_time")
    utterance_output_filename = ''
    if utterance_start_time is not None and utterance_end_time is not None and sentence_level_id is not None:
#      utterance_start_sample = int(round(utterance_start_time * sample_rate))
#      utterance_end_sample   = int(round(utterance_end_time * sample_rate))
      # Slice directly by samples
#      sentence_segment = audio_data[utterance_start_sample:utterance_end_sample, :] if audio_data.ndim > 1 else audio_data[utterance_start_sample:utterance_end_sample] # stereo or mono
      #Slice based on timestamps
      sentence_segment = audio_data[utterance_start_time * 1000:utterance_end_time*1000]
      utterance_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_sentence_segment.wav"
#      sf.write(sentence_segments_path + utterance_output_filename, sentence_segment, sample_rate)
      sentence_segment.export(sentence_segments_path + utterance_output_filename, format='wav')

  #  result = []
  #  annotator = ''
  #  try:
  #    result = annotations[1].get("result", [])
  #    annotator = annotations[1].get("completed_by").get("email")
  #  except:
  #    pass

    result = annotations[0].get("result", [])
    annotator = annotations[0].get("completed_by").get("email")

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
            category = category.replace(' ', '')
            if category in from_name:
              labels = value.get("choices", [])
              grouped[result_id]["error_labels"].extend(labels)

        if r_type == "textarea" and from_name == "spoken_words":
          category = grouped[result_id]["original_error_category"]
          if category in ["Run-on", "Contraction/Shortening"]:
          #  intended_words = ' '.join(value.get("text", []))
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
          #  intended_words = ' '.join(value.get("text", []))
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
          #  produced_word = ' '.join(value.get("text", []))
            grouped[result_id]["produced_word"].extend(value.get("text", []))

        if r_type == "textarea" and from_name == "mispronunciation_word":
          category = grouped[result_id]["original_error_category"]
          if category in [
            "Contraction/Shortening",
            "Other",
          #  "Unintelligible",
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

        ## e.g., "taxonomy": [["Disfluency Error", "Parental Aid"], ["Other"]]
        ## e.g., "taxonomy": [["Orthographic Sub.", "Phonological"], ["Phonological", "Consonant Substitution"]
        ## e.g., "taxonomy": [["Orthographic Sub.", "Phonological"], ["Phonological", "Consonant Substitution"], ["Phonological", "Consonant Omission"]

        if grouped[result_id]["original_error_category"] == "Mixed Error" and r_type == "taxonomy":
          category = grouped[result_id]["original_error_category"]
          if category == "Mixed Error":
            labels = []
            mixed = value.get("taxonomy", [])
            for tok in mixed:
              if tok[0] not in category:
                category = category + '+' + tok[0]
              if len(tok) > 1:
                labels = labels + tok[1 : ]
            grouped[result_id]["error_category"] = category
            grouped[result_id]["error_labels"] = labels

    # Create a row for each word
    word_segments = []
    for word_id, info in grouped.items():
        if utterance_output_filename != '' and info['error_labels'] != ["Unfilled Pause"] and info['start'] is not None and info['end'] is not None and sentence_level_id is not None: ## To remove lines with just comments

          ## Cleaning IPA
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
            elif intended_words == '' and produced_word == '' and IPA == []:
              pass


        #  if actual_production != '':
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

        #  if actual_production == '':
        #    print('EMPTY ACTUAL PRODUCTION:', info)

    # Sort by start time
    sorted_word_segments = sorted(word_segments, key=lambda x: (x["start"] is None, x["start"]))
    rows.extend(sorted_word_segments)

    # Collecting sentence-level segments and transcripts
    produced_utterance = []

    # Collecting word-level segments and transcripts
    if utterance_output_filename != '':
      sentence_audio_file = sentence_segments_path + utterance_output_filename    
      sentence_audio_data, sample_rate = sf.read(sentence_audio_file)
      start_time_list = []
      end_time_list = []
      first_word_start_time = 0
      for z in range(len(sorted_word_segments)):
        info = sorted_word_segments[z]      
        if info['start'] is not None and info['end'] is not None and sentence_level_id is not None: # and task_id in [195800258]: # [195798738, 195798730, 185499475, 185499476]: ## To remove lines with just comments   
          if z == 0:
            first_word_start_time = info['start']
          print(first_word_start_time)
          start_time = info['start'] - first_word_start_time
        #  if first_word_start_time > 1:
        #    start_time -= 1
          end_time = info['end'] - first_word_start_time
        #  if first_word_start_time > 1:
        #    end_time -=1
          start_time_list.append(start_time)
          end_time_list.append(end_time)
          actual_production = info['actual_production']                                                                                
          produced_utterance.append(actual_production)
          print(start_time, end_time)
          print(actual_production)
          # Convert to samples
          start_sample = int(round(start_time * sample_rate))
          end_sample   = int(round(end_time * sample_rate))
          print(start_sample, end_sample)
          # Slice directly by samples
          word_segment = sentence_audio_data[start_sample:end_sample, :] if sentence_audio_data.ndim > 1 else sentence_audio_data[start_sample:end_sample] # stereo or mono
          word_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_word_segment_{z+1}.wav"
          sf.write(word_segments_path + word_output_filename, word_segment, sample_rate)
          sf.write('test.wav', sentence_audio_data[0: int(round(1.1024390243902435*sample_rate))], sample_rate)

          word_segments_path_list.append(word_segments_path + word_output_filename)
          word_segments_transcript_list.append(actual_production)
          word_segments_intended_words_list.append(info['intended_words'])
          word_segments_produced_words_list.append(info['produced_word'])
          try:
            word_segments_IPA_list.append(''.join(info['IPA'][0]))
          except:
            word_segments_IPA_list.append('')
          word_segments_error_category_list.append(info['error_category'])
          word_segments_error_labels_list.append(info['error_labels'])

          ## Collecting n-gram word-level segments
          ngram_start_time = ''
          ngram_end_time = ''
          ngram_produced_utterance = ''
          if NGRAM == 'full':
            ngram_start_time = start_time_list[0]
            ngram_end_time = end_time
            ngram_produced_utterance = ' '.join(produced_utterance)
          else:
            ngram_start_time = start_time_list[-1 * NGRAM]
            ngram_end_time = end_time
            ngram_produced_utterance = ' '.join(produced_utterance[-1 * NGRAM : ])
          ngram_start_sample = int(round(ngram_start_time * sample_rate))
          ngram_end_sample = int(round(ngram_end_time * sample_rate))
          ngram_segment = sentence_audio_data[ngram_start_sample:ngram_end_sample, :] if sentence_audio_data.ndim > 1 else sentence_audio_data[ngram_start_sample:ngram_end_sample]
          ngram_output_filename = original_audio_name.split('.')[0] + f"_{task_id}_ngram_segment_{z+1}.wav"
          sf.write(word_segments_ngram_path + ngram_output_filename, ngram_segment, sample_rate)
          word_segments_ngram_path_list.append(word_segments_ngram_path + ngram_output_filename)
          word_segments_ngram_transcript_list.append(ngram_produced_utterance)

    if produced_utterance != []:
      sentence_segments_path_list.append(sentence_segments_path + utterance_output_filename)
      sentence_segments_transcript_list.append(' '.join(produced_utterance))
      sentence_segments_goldstandard_list.append(goldStandard)
      if original_audio_name not in story_audio_produced_sentences:
        story_audio_produced_sentences[original_audio_name] = [' '.join(produced_utterance)]
      else:
        story_audio_produced_sentences[original_audio_name].append(' '.join(produced_utterance))

    assert len(sentence_segments_transcript_list) == len(sentence_segments_path_list), "Mismatch in sentence segments and transcripts length"
    assert len(word_segments_path_list) == len(word_segments_transcript_list), "Mismatch in word segments and transcripts length"
    assert len(word_segments_path_list) == len(word_segments_error_category_list), "Mismatch in word segments and error categories length"
    assert len(word_segments_path_list) == len(word_segments_error_labels_list), "Mismatch in word segments and error labels length"
    assert len(word_segments_ngram_path_list) == len(word_segments_ngram_transcript_list), "Mismatch in n-gram word segments and transcripts length"
    assert len(word_segments_ngram_path_list) == len(word_segments_error_category_list), "Mismatch in n-gram word segments and word segments length"

    ## -------- Checking for segmentation overlap --------
    overlap_found = False
    for i in range(len(sorted_word_segments) - 1):
      task_id = sorted_word_segments[i]["task_id"]
      annotator = sorted_word_segments[i]["annotator"]
      current_info = sorted_word_segments[i]
      next_info = sorted_word_segments[i+1]
      if current_info["end"] is not None and next_info["start"] is not None:
          if current_info["end"] > next_info["start"]:
              error_dict[annotator].append([task_id, annotator, "Overlap detected between the following", current_info, next_info, '\n'])
              overlap_found = True

# Generating story-level datasets for modeling
for audio, sentences in story_audio_goldstandard_sentences.items():
   story_data_path_list.append('/Users/liu.ying/University of Florida/Leite,Walter - Storiza Corpus Spring 2025/Storiza_Participant_Recording_05-30-25/' + audio)
   story_data_goldstandard_list.append(' '.join(sentences))
   story_data_transcript_list.append(' '.join(story_audio_produced_sentences[audio]))

story_data['Path'] = story_data_path_list
story_data['Transcript'] = story_data_transcript_list
story_data['goldStandard'] = story_data_goldstandard_list
story_data.to_csv('../processed_data/story_level_data.csv', index=False, encoding="utf-8")

# Generating sentence-level datasets for modeling
sentence_segments_data['Path'] = sentence_segments_path_list
sentence_segments_data['Transcript'] = sentence_segments_transcript_list
sentence_segments_data['goldStandard'] = sentence_segments_goldstandard_list
sentence_segments_data.to_csv('../processed_data/sentence_level_data.csv', index=False, encoding="utf-8")

# Generating word-level datasets for modeling
word_segments_data['Path'] = word_segments_path_list
word_segments_data['Transcript'] = word_segments_transcript_list
word_segments_data['Intended Words'] = word_segments_intended_words_list
word_segments_data['Produced Words'] = word_segments_produced_words_list
word_segments_data['IPA'] = word_segments_IPA_list
word_segments_data['Error Category'] = word_segments_error_category_list
word_segments_data['Error Labels'] = word_segments_error_labels_list
word_segments_data.to_csv('../processed_data/word_level_data.csv', index=False, encoding="utf-8")

word_segments_data_ngram['Path'] = word_segments_ngram_path_list
word_segments_data_ngram['Transcript'] = word_segments_ngram_transcript_list
word_segments_data_ngram['Intended Words'] = word_segments_intended_words_list
word_segments_data_ngram['Produced Words'] = word_segments_produced_words_list
word_segments_data_ngram['IPA'] = word_segments_IPA_list
word_segments_data_ngram['Error Category'] = word_segments_error_category_list
word_segments_data_ngram['Error Labels'] = word_segments_error_labels_list
word_segments_data_ngram.to_csv('../processed_data/word_level_data_ngram.csv', index=False, encoding="utf-8")

# -------- Write Full word-level data --------
csv_path = "../../processed_data/formatted_annotations_with_comments.csv"
task_intended_words_dict = {}

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["task_id", "annotator", "start", "end", "intended_words", "produced_word", "IPA", "error_category", "error_labels", "comments", "goldStandard", "original_audio_name", "actual_production"])
    writer.writeheader()
    for row in rows:   
        comments = row.get("comments")
        start = row.get("start")
        end = row.get("end")
        ## Doing checks
     #   if comments == '': # for each task, if there are comments, comments together are assigned a new id, and there are no other annotations under this id
        if start is not None and end is not None:
          task_id = row.get("task_id")      
          annotator = row.get("annotator")
          intended_words = row.get("intended_words")
          produced_word = row.get("produced_word")
          IPA = row.get("IPA")
          category = row.get("error_category")
          labels = row.get("error_labels")
          goldStandard = row.get("goldStandard")

          # Keeping track of all intended words written by annotators to identify if anything is misspelled
          if task_id not in task_intended_words_dict:
            task_intended_words_dict[task_id] = [intended_words]
          else:
            task_intended_words_dict[task_id].append(intended_words)

          new_row = row
          del new_row["task_id"]
          del new_row["annotator"]
          del new_row["start"]
          del new_row["end"]
          del new_row["comments"]

          ## Checking to see if annotators misspelled any word
          translator = str.maketrans('', '', string.punctuation)
          modified_goldStandard = [item.translate(translator) for item in goldStandard.split()]

          if "Self Response" not in category and "Whispering" not in category and "Word Insertion" not in labels and "Unfilled Pause" not in labels and intended_words != "Let's":
            if 'Contraction' in category and intended_words not in goldStandard:
              error_dict[annotator].append([task_id, annotator, "Intended word(s) might be misspelled", new_row])
            if 'Run-on' in category and ' ' in intended_words and intended_words not in goldStandard:
              error_dict[annotator].append([task_id, annotator, "Intended word(s) might be misspelled", new_row])
            if intended_words != '' and 'Contraction' not in category and 'Run-on' not in category:
              if intended_words not in modified_goldStandard and intended_words not in goldStandard.split():
                error_dict[annotator].append([task_id, annotator, "Intended word(s) might be misspelled", new_row])
              else:
                if intended_words[0].isupper() is False and intended_words.capitalize() in modified_goldStandard and intended_words not in goldStandard.split():
                  if intended_words.capitalize() not in task_intended_words_dict[task_id]:
                    if [task_id, annotator, "NEW Intended word(s) might be misspelled", comments, new_row] not in error_dict[annotator]:
                      error_dict[annotator].append([task_id, annotator, "NEW Intended word(s) might be misspelled", comments, new_row])

          if produced_word == "NONE":
            error_dict[annotator].append([task_id, annotator, "Don't write NONE as Produced Word", new_row])      

          if category == '':
            error_dict[annotator].append([task_id, annotator, "Missing Error Category", new_row])

          if category not in ["Correct", "Contraction/Shortening", "Unintelligible", "Whispering", "Other", "Self Response", "Run-on"] and len(labels) == 0:
            if "Mixed Error" not in category:
              error_dict[annotator].append([task_id, annotator, "Missing choosing Error Type(s)", new_row])  
            else:
               exclude_count = 1
               category_list = category.split('+')
               for c in ["Correct", "Contraction/Shortening", "Unintelligible", "Whispering", "Other", "Self Response", "Run-on"]:
                  if c in category_list or c in category:
                    exclude_count += 1
               if exclude_count != len(category_list):
                  error_dict[annotator].append([task_id, annotator, "Missing choosing Error Type(s)", new_row])

          if 'Run-on' in category or 'Run-on' in labels:
            if ' ' in intended_words:
              error_dict[annotator].append([task_id, annotator, "The run-on word needs to be segmented if possible", new_row])

          if "Mixed Error" in category and len(category.split("+")) <= 1:
            error_dict[annotator].append([task_id, annotator, "Only one error label was found for Mixed Error", new_row]) 

          ### Checking Maddie's previous annotations
          if "Prolongation" in labels: # and category != "Disfluency Error":
            # Maddie's previous annotations
            # IPA: : --> Consistent with guideline
            # Produced Word: Repetition of the prolonged sound
            try:
              if 'ː' in IPA[0]:
                IPA[0] = IPA[0].replace('ː', ':')
                row['IPA'] = IPA
              if ":" not in IPA[0]: # and 'ː' not in IPA[0]:
                error_dict[annotator].append([task_id, annotator, "Might be missing : for IPA for Prolongation", new_row]) 
            except:
              pass
            #  error_dict[annotator].append([task_id, annotator, "Might be missing IPA", new_row]) 

            if produced_word != '':
              if ":" not in produced_word and 'ː' not in produced_word:
                error_dict[annotator].append([task_id, annotator, "Might need to add : for Produced Word for Prolongation", new_row]) 

          if "Stutter" in labels:
            # Maddie's annotations
            # IPA: . --> Consistent with guideline
            # Produced Word: A repetition of the stuttered part and three dots … (Ex: h…hi)
            try:
              if "." not in IPA[0]:
                error_dict[annotator].append([task_id, annotator, "Might be missing . for IPA for Stutter", new_row]) 
            except:
              pass
            #  error_dict[annotator].append([task_id, annotator, "Might be missing IPA", new_row]) 

            if produced_word != '':               
                if "." not in produced_word:
                   error_dict[annotator].append([task_id, annotator, "Might be missing . for Produced Word for Stutter", new_row]) 

          if "Broken Word" in labels:
            # Maddie's previous annotations:    
            # IPA: .
            # Produced Word: three dots …
            # Resolving inconsistencies in Maddie's annotations
            if "rose" in annotator and "Stutter" not in labels:
              if '...' in IPA[0]:
                IPA[0] = IPA[0].replace('...', '|')
                row['IPA'] = IPA
              elif '..' in IPA[0]:
                IPA[0] = IPA[0].replace('..', '|')
                row['IPA'] = IPA
              elif '.' in IPA[0]:
                IPA[0] = IPA[0].replace('.', '|')
                row['IPA'] = IPA

              if produced_word != '':
                if '...' in produced_word:
                  produced_word = produced_word.replace('...', '|')
                  row['produced_word'] = produced_word
                elif '..' in produced_word:
                  produced_word = produced_word.replace('..', '|')
                  row['produced_word'] = produced_word
                elif '.' in produced_word:
                  produced_word = produced_word.replace('.', '|')
                  row['produced_word'] = produced_word

            # Uncomment after Maddie fixes her annotations
          #  if "Stutter" not in labels:
          #    assert '.' not in IPA[0]

            if "rose" in annotator and "Stutter" not in labels and "|" not in IPA[0]: 
              error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit IPA for Broken Word", new_row])  
            if "rose" in annotator and "Stutter" in labels and "|" not in IPA[0]: ## Check to see if Maddie has fixed these
              error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit IPA for Broken Word and Stutter", new_row])  

            try:
              if "rose" not in annotator and "|" not in IPA[0]:             
                error_dict[annotator].append([task_id, annotator, "Might be missing | for IPA for Broken Word", new_row])     
            except:
              error_dict[annotator].append([task_id, annotator, "Might be missing IPA", new_row])     

            if produced_word != '':
              if "rose" in annotator and "Stutter" not in labels and "|" not in produced_word: 
                error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit Produced Word for Broken Word", new_row])             
              if "rose" in annotator and "Stutter" in labels and "|" not in produced_word: ## Check to see if Maddie has fixed these
                error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit Produced Word for Broken Word and Stutter", new_row])  

            #  if "rose" in annotator and "." not in produced_word and "|" not in produced_word:
            #    error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit Produced Word for Broken Word", new_row])  
            #  if "rose" in annotator and "." in produced_word and "Stutter" in labels: ## Check to see if Maddie has fixed these
            #    error_dict[annotator].append([task_id, annotator, "Maddie might need to revisit Produced Word for Broken Word and Stutter", new_row])  

              try:
                if "rose" not in annotator and "|" not in produced_word:             
                  error_dict[annotator].append([task_id, annotator, "Might be missing | for Produced Word for Broken Word", new_row])  
              except:
                error_dict[annotator].append([task_id, annotator, "Might be missing Produced Word", new_row])  

        #  if "Visual" in category:
        #     error_dict[annotator].append([task_id, annotator, "Check your annotations for Visual Tracking Errors", new_row])  

          if IPA == []:
            if "Contraction/Shortening" in category or "Phonological Error" in category or "Orthographic Error" in category or "Grammatical Error" in category or (category == "Structural Error" and "Word Insertion" in labels) or (category == "Disfluency Error" and "Stutter" in labels) or (category == "Disfluency Error" and "Interjection" in labels) or (category == "Disfluency Error" and "Prolongation" in labels) or (category == "Disfluency Error" and "Broken Word" in labels):
              error_dict[annotator].append([task_id, annotator, "Missing IPA", new_row])

          if intended_words == '' and produced_word == '' and IPA == []:
            if labels != ["Unfilled Pause"]:
              if category not in ["Unintelligible", "Whispering", "Other"] and "Mixed Error" not in category:
                if not (category == "Disfluency Error" and labels == []):
                  error_dict[annotator].append([task_id, annotator, "Need to provide relevant annotations for this word; especially if the Error Type contains Parental Aid, make sure to re-consult the guideline", new_row]) 

              elif "Mixed Error" in category:
                exclude_count = 1
                category_list = category.split('+')
                for c in ["Unintelligible", "Whispering", "Other"]:
                  if c in category_list or c in category:
                    exclude_count += 1
                if exclude_count != len(category_list):
                  error_dict[annotator].append([task_id, annotator, "Need to provide relevant annotations for this word; especially if the Error Type contains Parental Aid, make sure to re-consult the guideline", new_row])

          if not (intended_words == '' and produced_word == '' and IPA == []):
            if labels == ["Parental Aid"] and produced_word == '':
              try:
                assert intended_words != ''
              except:
                error_dict[annotator].append([task_id, annotator, "Consult the guideline about your annotations for Parental Aid and Error Category", new_row])

          if not (intended_words == '' and produced_word == '' and IPA == []):
            if intended_words == '':
              if labels == ["Parental Aid"]: 
                try:
                  assert "Mixed Error" in category and "Other" in category
                  assert produced_word != ''
                except:
                  error_dict[annotator].append([task_id, annotator, "Consult the guideline about your annotations for Parental Aid and Error Category", new_row])

              elif labels != ['Unfilled Pause'] and "Interjection" not in labels and "Word Insertion" not in labels:
                exclude = ["Whispering", "Other", "Unintelligible", "Self Response"]
                if category not in exclude:
                  if "Mixed Error" in category:
                    exclude_count = 1
                    category_list = category.split('+')
                    for c in exclude:
                      if c in category_list:
                        exclude_count += 1
                    if exclude_count != len(category_list):
                      #  if produced_word == '':
                      intended_word_check = 'not pass'
                      error_dict[annotator].append([task_id, annotator, "Missing intended word(s)", new_row, exclude_count])

                  else:
                  #  if produced_word == '':
                    intended_word_check = 'not pass'
                    error_dict[annotator].append([task_id, annotator, "Missing intended word(s)", new_row])

          if ("Contraction/Shortening" in category or "Visual Tracking Error" in category or "Self Response" in category or "Orthographic Error" in category or "Grammatical Error" in category or "Structural Error" in category or "Other" in category):
          #  if "Mixed Error" not in category:
            if produced_word == '':
              error_dict[annotator].append([task_id, annotator, "Missing produced word(s)", new_row])

        #  if "Disfluency" in category and "Unfilled Pause" not in labels and labels != [] and labels != ["Interjection"] and intended_words == '' and produced_word == '':
        #    print("CHECK ME", task_id, annotator, new_row)

        #  if "Other" in category and intended_words == '' and produced_word == '':
        #    print(task_id, annotator, "CHECK OTHER", new_row)

          if "Unfilled Pause" in labels and intended_words != '':
            error_dict[annotator].append([task_id, annotator, "There should not be Intended Word filled out for Unfilled Pause", new_row])

          if "Word Insertion" in labels and intended_words != '':
            error_dict[annotator].append([task_id, annotator, "There should not be Intended Word filled out for Word Insertion", new_row])

          if "Interjection" in labels and intended_words != '':
            error_dict[annotator].append([task_id, annotator, "There should not be Intended Word filled out for Interjection", new_row])

          if "Word Omission" in labels:
            error_dict[annotator].append([task_id, annotator, "Per guideline, you do not need to annotate Word Omission", new_row])

          if IPA != [] and "." in IPA[0] and "Stutter" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have . for the IPA, then need to add Stutter in the label", new_row])
          if "." in produced_word and "Stutter" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have . for the Produced Word, then need to add Stutter in the label", new_row])

          if IPA != [] and ":" in IPA[0] and "Prolongation" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have : for the IPA, then need to add Prolongation in the label", new_row])
          if ":" in produced_word and "Prolongation" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have : for the Produced Word, then need to add Prolongation in the label", new_row])

          if IPA != [] and "|" in IPA[0] and "Broken Word" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have | for the IPA, then need to add Broken Word in the label", new_row])
          if "|" in produced_word and "Broken Word" not in labels and labels != ["Parental Aid"]:
            error_dict[annotator].append([task_id, annotator, "If you have | for the Produced Word, then need to add Broken Word in the label", new_row])

          if annotator in ["claire.kuntz@ufl.edu", "ghineae@ufl.edu", "hamern@ufl.edu", "katherine.ball@ufl.edu", "benjaminmcneill@ufl.edu"] and "Unfilled Pause" in labels:
            error_dict[annotator].append([task_id, annotator, "Per guideline, you do not need to annotate Unfilled Pause; please remove them", new_row])

          row['start'] = str(start)
          row['end'] = str(end)
          writer.writerow(row)

print(f"✅ Done! Output saved to: {csv_path}")

for annotator in annotator_list:
  with open('../fixes/' + annotator_map[annotator] + '_fixes.txt', 'w') as f:
    annotation_errors = error_dict[annotator]
    try:
      sorted_annotation_errors = sorted(annotation_errors, key=lambda x: x[0])
    except:
      print(sorted_annotation_errors)

    for error in sorted_annotation_errors:
      f.write(' '.join(str(tok) for tok in error) + '\n')

  f.close()


# In[ ]:





# In[ ]:


### Checking cross-annotations; JSON file seems to only save the latest annotations (from two annotators)
import pandas as pd

csv_path = pd.read_csv("export_157618_project-157618-at-2025-08-04-03-01-f5029ec4.csv")
word_annotation_list = csv_path['WordAnnotation'].tolist()
id_list = csv_path['id'].tolist()
id_annotations_dict = {}
for i in range(len(id_list)):
    idx = id_list[i]
    annotation = json.loads(word_annotation_list[i])
    annotation_list = []
    for tok in annotation:
        for k, v in tok.items():
            annotation_list.append(k + ' ' + str(v))
    if idx not in id_annotations_dict:       
        id_annotations_dict[idx] = [annotation_list]
    else:
        id_annotations_dict[idx].append(annotation_list)

overall_num_segment_agreement = []
overall_error_category_agreement = []
overall_error_label_agreement = []

for idx in list(set(id_list)):
    if id_list.count(idx) > 1:
        print(idx)
        max_len = []
        for tok in id_annotations_dict[idx]:
            tok_len = len(tok)
            max_len.append(tok_len)
        max_len = max(max_len)
        annotation1 = id_annotations_dict[idx][0]
        new_annotation1 = []
        for ann in annotation1:
            if 'Unfilled Pause' not in ann:
                new_annotation1.append(ann)

        annotation2 = id_annotations_dict[idx][1]
        new_annotation2 = []
        for ann in annotation2:
            if 'Unfilled Pause' not in ann:
                new_annotation2.append(ann)

        if len(new_annotation1) == len(new_annotation2):
            overall_num_segment_agreement.append(1)
        else:
            overall_num_segment_agreement.append(0)

        for i in range(max_len):
            try:
                print(new_annotation1[i], new_annotation2[i])
            except:
                try:
                    print('NONE', new_annotation2[i])
                except:
                    print(new_annotation1[i], 'NONE')
    #    for tok in id_annotations_dict[idx]:
    #    print(tok)


