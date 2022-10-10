"""
This is the config file of Digit Recognizer
"""
import os
import pathlib

from cake_classifier.models.individual_pretrained import Pretrained
from cake_classifier.models.combined_pretrained import Combined_Model

SEED = 1000

#---PATH---
MAIN_PATH = pathlib.Path(__file__).resolve().parents[2] #cv-assignment
print(MAIN_PATH)
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH/ "model_checkpoint"

MODEL_PARAMS = {
    # "pretrained_freeze": {"model": Pretrained, "model_name": "resnet", "freeze": True},

    "combined_freeze": {"model": Combined_Model, "model_name": ["resnet", "mobile"], "freeze": True}

}

