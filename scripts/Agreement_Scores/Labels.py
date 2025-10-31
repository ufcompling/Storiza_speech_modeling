from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

# ------------------------------
# Constants from labeling UI
# ------------------------------

TOP_LEVEL_LABELS = [
    "Correct",
    "Phonological Error",
    "Orthographic Error",
    "Grammatical Error",
    "Structural Error",
    "Visual Tracking Error",
    "Disfluency Error",
    "Whispering",
    "Run-on",
    "Contraction/Shortening",
    "Unintelligible",
    "Self Response",
    "Other",
]

TOP_LEVEL_ERROR_LABELS = [
    "Phonological Error",
    "Orthographic Error",
    "Grammatical Error",
    "Structural Error",
    "Visual Tracking Error",
    "Disfluency Error",
    "Whispering",
    "Run-on",
    "Contraction/Shortening",
    "Unintelligible",
    "Self Response",
    "Other",
]

# Specific (error-type) labels
PHONO_SPEC = [
    "Consonant Substitution",
    "Vowel Substitution",
    "Consonant Omission",
    "Vowel Omission",
    "Consonant Insertion",
    "Vowel Insertion",
    "Misplaced Stress",
]

ORTHO_SPEC = [
    "Letter Reversal",
    "Left Right Tracking Substitution",
    "Phonological Substitution",
    "Contextual Substitution",
    "Unrelated Substitution",
]

GRAM_SPEC = ["Omission", "Substitution", "Insertion"]

STRUCT_SPEC = ["Word Omission", "Word Insertion"]

VISUAL_SPEC = ["Skip Line", "Backtrack", "Wrong Order"]

DISFLUENCY_SPEC = [
    "Stutter",
    "Word Repetition",
    "Repair",
    "Broken Word",
    "Prolongation",
    "Interjection",
    "Parent Correction",
    "Parental Aid",
]

# Present only under Mixed taxonomy (leaf); include as a specific for completeness
RUNON_SPEC = ["Run-on Word"]

SPECIFIC_LABELS = (
    ORTHO_SPEC
    + PHONO_SPEC
    + GRAM_SPEC
    + STRUCT_SPEC
    + VISUAL_SPEC
    + DISFLUENCY_SPEC
)

# Mapping from MixedErrorTaxonomy families to top-level labels
TAXONOMY_TO_TOPLEVEL = {
    "Orthographic Sub.": "Orthographic Error",
    "Phonological": "Phonological Error",
    "Grammatical": "Grammatical Error",
    "Structural": "Structural Error",
    "Visual Tracking": "Visual Tracking Error",
    "Disfluency Error": "Disfluency Error",
    "Run-on Word": "Run-on",  # leaf acts as family here
    "Whispering": "Whispering",  # leaf
    "Contraction/Shortening": "Contraction/Shortening",  # leaf
    "Other": "Other",
}

# From_name -> which specific label inventory to read
FROMNAME_TO_SPECINV = {
    "PhonologicalErrorType": PHONO_SPEC,
    "OrthographicErrorType": ORTHO_SPEC,
    "GrammaticalErrorType": GRAM_SPEC,
    "StructuralErrorType": STRUCT_SPEC,
    "VisualTrackingErrorType": VISUAL_SPEC,
    "DisfluencyErrorType": DISFLUENCY_SPEC,
}


# ---- mapping from specific -> parent top-level ----
SPECIFIC_TO_TOPLEVEL: Dict[str, str] = {
    # Phonological
    "Consonant Substitution": "Phonological Error",
    "Vowel Substitution": "Phonological Error",
    "Consonant Omission": "Phonological Error",
    "Vowel Omission": "Phonological Error",
    "Consonant Insertion": "Phonological Error",
    "Vowel Insertion": "Phonological Error",
    "Misplaced Stress": "Phonological Error",

    # Orthographic
    "Letter Reversal": "Orthographic Error",
    "Left Right Tracking Substitution": "Orthographic Error",
    "Phonological Substitution": "Orthographic Error",
    "Contextual Substitution": "Orthographic Error",
    "Unrelated Substitution": "Orthographic Error",

    # Grammatical
    "Omission": "Grammatical Error",
    "Substitution": "Grammatical Error",
    "Insertion": "Grammatical Error",

    # Structural
    "Word Omission": "Structural Error",
    "Word Insertion": "Structural Error",

    # Visual Tracking
    "Skip Line": "Visual Tracking Error",
    "Backtrack": "Visual Tracking Error",
    "Wrong Order": "Visual Tracking Error",

    # Disfluency
    "Stutter": "Disfluency Error",
    "Word Repetition": "Disfluency Error",
    "Repair": "Disfluency Error",   # aka 'Self Correction'
    "Broken Word": "Disfluency Error",
    "Prolongation": "Disfluency Error",
    "Interjection": "Disfluency Error",
    "Parent Correction": "Disfluency Error",
    "Parental Aid": "Disfluency Error",
    "Run-on Word": "Disfluency Error",
}

NAME_NORMALIZING_MAP={
    "Letter Reversal Substitution": "Letter Reversal"
}

# 1. Initial/Grouped Combos
FULL_COMBOS = [
    ("Top-level (general_label_type)", TOP_LEVEL_LABELS, "general_label_type"),
    ("Top-level (not including correct)", TOP_LEVEL_ERROR_LABELS, "general_label_type"),
    #All Specific label combined toghether
    ("Phonological specifics combined", PHONO_SPEC, "specific_label_type"),
    ("Orthographic specifics combined", ORTHO_SPEC, "specific_label_type"),
    ("Grammatical specifics combined", GRAM_SPEC, "specific_label_type"),
    ("Structural specifics combined", STRUCT_SPEC, "specific_label_type"),
    ("Visual specifics combined", VISUAL_SPEC, "specific_label_type"),
    ("Disfluency specifics combined", DISFLUENCY_SPEC, "specific_label_type"),
    ("All specifics combined", SPECIFIC_LABELS, "specific_label_type"),
    # Non-label attributes
    ("Intended word", ["intended_word"], None),
    ("Produced word", ["produced_word"], None),
    ("Mispronunciation IPA", ["mispronunciation_ipa"], None),
]

# 2. Add individual Top-Level Labels
for label in TOP_LEVEL_LABELS:
    # Use a clear name for the individual label combo
    combo_name = f"Individual Top: {label}"
    FULL_COMBOS.append((combo_name, [label], "general_label_type"))

# 3. Add individual specifics combined
for label in PHONO_SPEC:
    # Use a clear name for the individual phonological specific combo
    combo_name = f"Individual Phono Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))

# Orthographic
for label in ORTHO_SPEC:
    combo_name = f"Individual Ortho Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))

# Grammatical
for label in GRAM_SPEC:
    combo_name = f"Individual Gram Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))

# Structural
for label in STRUCT_SPEC:
    combo_name = f"Individual Struct Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))

# Visual Tracking
for label in VISUAL_SPEC:
    combo_name = f"Individual Visual Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))

# Disfluency
for label in DISFLUENCY_SPEC:
    combo_name = f"Individual Disfluency Spec: {label}"
    FULL_COMBOS.append((combo_name, [label], "specific_label_type"))
