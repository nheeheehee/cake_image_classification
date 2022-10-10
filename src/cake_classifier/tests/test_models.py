""" This module is to test model
"""

import torch
from cake_classifier.config import MODEL_PARAMS
from copy import deepcopy

def test_model():
    """ Test all models based on settings per config.py for correct output size"""

    for model_ in MODEL_PARAMS:
        model = MODEL_PARAMS[model_]["model"](model_name = MODEL_PARAMS[model_]["model_name"])
        sample = torch.rand(32, 3, 224, 224)

        x = model(sample)
        assert x.size() == torch.Size([32,5]), f"Output mismatched for {model_}"
 
if __name__ == "__main__":
    test_model()


# def is_freeze():
#     """Test if the pretrained layers are freeze i.e. weights before and after backprop
#     are the same"""

#     for model_name in MODEL_PARAMS:
#         model = MODEL_PARAMS[model_name]["model"](freeze = MODEL_PARAMS[model_name]["freeze"])
#         if model_name == "pretrained":
#             weights = deepcopy(list(model.parameters())[:-6])
#             sample = torch.rand(32,3,224,224)
#             x = model(sample)
#             updated_weights = list(x)



