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
    "Letter Reversal Substitution",
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
    "Self Correction",
    "Broken Word",
    "Prolongation",
    "Interjection",
    "Unfilled Pause",
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
    + RUNON_SPEC
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
    "Letter Reversal Substitution": "Orthographic Error",
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
    "Self Correction": "Disfluency Error",   # aka 'Repair'
    "Broken Word": "Disfluency Error",
    "Prolongation": "Disfluency Error",
    "Interjection": "Disfluency Error",
    "Unfilled Pause": "Disfluency Error",
    "Parent Correction": "Disfluency Error",
    "Parental Aid": "Disfluency Error",
    "Run-on Word": "Disfluency Error",
}