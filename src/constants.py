""""
Author: Md Mostafizur Rahman
File: Configaration file
"""

import os
from pathlib import Path
from typing import List

from typing_extensions import Literal

# directory-related
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
SRC = ROOT / "src"
OUT = ROOT / "output"
PLOT_CORR_REG_CLASS_OUTPUT_PATH = ROOT / "reg_class_correlation_plots"
if not OUT.exists():
    os.makedirs(OUT, exist_ok=True)
if not PLOT_CORR_REG_CLASS_OUTPUT_PATH.exists():
    os.makedirs(PLOT_CORR_REG_CLASS_OUTPUT_PATH, exist_ok=True)

# names and types
DATASET_NAMES = ["Parkinsons", "Diabetes", "Liver"]
ECMethod = Literal["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                                "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]
EC_METHODS: List[ECMethod] = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                                "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]

# analysis constants
SEED = 42
TEST_SIZE = 0.2
