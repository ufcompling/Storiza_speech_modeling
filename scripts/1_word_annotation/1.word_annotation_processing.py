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
from dataHelpers import *

# ---- Processing parameters ----
SAMPLE_RATE = 4800
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
sentenceLabels_original_audio_timestamps = load_sentence_label_timestamps(SENTENCE_LABELS_CSV) #WTW why do we even need this? we can just directly use the audio timestamps


if DEBUG:

    print_structure_with_values(list(sentenceLabels_original_audio_timestamps)[:min(5, len(sentenceLabels_original_audio_timestamps.keys()))])

    #try specific example:
    try:
        print(sentenceLabels_original_audio_timestamps[""])
    except KeyError:
        print("item not found")


#============= (3) Collecting annotated intended/goldstandard sentences for each audio================

story_audio_goldstandard_sentences = defaultdict(list)
story_audio_produced_sentences = defaultdict(list)

# Story-level data
story_data = pd.DataFrame(columns=['Path', 'Transcript', 'goldStandard'])
# Sentence-level data
sentence_segments_data = pd.DataFrame(columns=['Path', 'Transcript', 'goldStandard']) 


try:
    os.makedirs(SENTENCE_SEGMENTS_DIR, exist_ok=True)
    print(f"Directory '{SENTENCE_SEGMENTS_DIR}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{SENTENCE_SEGMENTS_DIR}': {e}")

# Unigram word-level data
word_segments_data = pd.DataFrame(columns=['Path', 'Transcript', 'Intended Words', 'Produced Words', 'IPA', 'Error Category', 'Error Labels'])
try:
    os.makedirs(WORD_SEGMENTS_DIR, exist_ok=True)
    print(f"Directory '{WORD_SEGMENTS_DIR}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{WORD_SEGMENTS_DIR}': {e}")

# N-gram word-level data, taking context into account when doing classification later
word_segments_data_ngram = pd.DataFrame(columns=['Path', 'Transcript', 'Intended Words', 'Produced Words', 'IPA', 'Error Category', 'Error Labels'])
try:
    os.makedirs(WORD_SEGMENTS_NGRAM_DIR, exist_ok=True)
    print(f"Directory '{WORD_SEGMENTS_NGRAM_DIR}' created or already exists.")
except OSError as e:
    print(f"Error creating directory '{WORD_SEGMENTS_NGRAM_DIR}': {e}")



error_dict = {}
for annotator in ANNOTATOR_LIST:
  error_dict[annotator] = []
#============= (4) Extract & Format Data================

(
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
) = build_segments_and_rows(
    data=data,
    audio_base_dir=AUDIO_BASE_DIR,
    SENTENCE_SEGMENTS_DIR=SENTENCE_SEGMENTS_DIR,
    WORD_SEGMENTS_DIR=WORD_SEGMENTS_DIR,
    WORD_SEGMENTS_NGRAM_DIR=WORD_SEGMENTS_NGRAM_DIR,
    BUFFERED_UTTERANCES_SUBDIR=BUFFERED_UTTERANCES_SUBDIR,
    sentenceLabels_original_audio_timestamps=sentenceLabels_original_audio_timestamps,
    NGRAM=NGRAM,
    error_dict=error_dict,  # can be a defaultdict(list) or a seeded dict
    sample_rate=SAMPLE_RATE,
    debug=DEBUG
)


# Generating story-level datasets for modeling
story_data_path_list=[]
story_data_goldstandard_list=[]
story_data_transcript_list=[]
for audio, sentences in story_audio_goldstandard_sentences.items():
   story_data_path_list.append(os.path.join(AUDIO_BASE_DIR,audio))
   story_data_goldstandard_list.append(' '.join(sentences))
   story_data_transcript_list.append(' '.join(story_audio_produced_sentences[audio]))

story_data['Path'] = story_data_path_list
story_data['Transcript'] = story_data_transcript_list
story_data['goldStandard'] = story_data_goldstandard_list
story_data.to_csv(STORY_LEVEL_CSV, index=False, encoding="utf-8")

# Generating sentence-level datasets for modeling
sentence_segments_data['Path'] = sentence_segments_path_list
sentence_segments_data['Transcript'] = sentence_segments_transcript_list
sentence_segments_data['goldStandard'] = sentence_segments_goldstandard_list
sentence_segments_data.to_csv(SENTENCE_LEVEL_CSV, index=False, encoding="utf-8")

# Generating word-level datasets for modeling
word_segments_data['Path'] = word_segments_path_list
word_segments_data['Transcript'] = word_segments_transcript_list
word_segments_data['Intended Words'] = word_segments_intended_words_list
word_segments_data['Produced Words'] = word_segments_produced_words_list
word_segments_data['IPA'] = word_segments_IPA_list
word_segments_data['Error Category'] = word_segments_error_category_list
word_segments_data['Error Labels'] = word_segments_error_labels_list
word_segments_data.to_csv(WORD_LEVEL_CSV, index=False, encoding="utf-8")

word_segments_data_ngram['Path'] = word_segments_ngram_path_list
word_segments_data_ngram['Transcript'] = word_segments_ngram_transcript_list
word_segments_data_ngram['Intended Words'] = word_segments_intended_words_list
word_segments_data_ngram['Produced Words'] = word_segments_produced_words_list
word_segments_data_ngram['IPA'] = word_segments_IPA_list
word_segments_data_ngram['Error Category'] = word_segments_error_category_list
word_segments_data_ngram['Error Labels'] = word_segments_error_labels_list
word_segments_data_ngram.to_csv(WORD_LEVEL_NGRAM_CSV, index=False, encoding="utf-8")

# -------- Write Full word-level data --------
task_intended_words_dict = {}

with open(FORMATTED_FULL_CSV, "w", newline="", encoding="utf-8") as f: #bruh, 400 lines
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

print(f"✅ Done! Output saved to: {FORMATTED_FULL_CSV}")

exit()
# IDK what is happening here
for annotator in ANNOTATOR_LIST:
    #I DONT HAVE THESE FILES :((((((((((((((((((((((((((((
  with open(FIXES_DIR + ANNOTATOR_MAP[annotator] + '_fixes.txt', 'w') as f:
    print(FIXES_DIR + ANNOTATOR_MAP[annotator] + '_fixes.txt',)
    annotation_errors = error_dict[annotator]
    try:
      sorted_annotation_errors = sorted(annotation_errors, key=lambda x: x[0])
    except:
      print(sorted_annotation_errors)

    for error in sorted_annotation_errors:
      f.write(' '.join(str(tok) for tok in error) + '\n')

  f.close()




### Checking cross-annotations; JSON file seems to only save the latest annotations (from two annotators)
import pandas as pd

csv_path = pd.read_csv(CROSS_ANNOT_CSV)
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


