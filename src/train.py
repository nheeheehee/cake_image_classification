import argparse
from ast import arg 
import os
from pyparsing import lru_cache 
import torch

from cake_classifier.config import MODEL_PARAMS, MODEL_PATH
from cake_classifier.dataset.make_dataset import DataManager
from cake_classifier.utils import seed_everything
from cake_classifier.classifier import Classifier

def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.

    Args:
        - Model 
        - No of epoch
        - Learning rate
        - Batch size 
    
    Returns:
        argparse.ArgumentParser().parse_args()

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help = "choice of model and whether to freeze the pretrained layers (default: pretrained_freeze).\
                If an integer is specified with the model -> number of layers being froze (e.g. pretrained_30)",
        type = str,
        default = "pretrained_freeze"
    )

    # parser.add_argument(
    #     "-f",
    #     "--freeze",
    #     help = "choice of whether to freeze the pretrained layers",
    #     type = str,
    #     default = "True"
    # )

    parser.add_argument(
        "-e",
        "--epochs",
        help = "number of epochs (default: 1)",
        type = int,
        default = 1
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help = "Number of images in a batch (default: 32)",
        type = int,
        default = 32
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help = "learning rate for optimizer (default: 1e-4)",
        type = int, 
        default = 1e-4
    )

    parser.add_argument(
        "-v",
        "--version",
        help = "checkpoint location to load from (default: -1 (No checkpoint load))",
        type = int,
        default = -1
    )

    args = parser.parse_args()
    return args 

def main():
    """This is for training the model"""

    args = get_argument_parser()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    model_name = args.model
    ver = args.version

    seed_everything()

    print(
        f"No of epochs: {epochs} \n Batch size: {batch_size} \n Learning rate: {lr}"
    )

    datamodule = DataManager(batch_size = batch_size)

    if model_name in MODEL_PARAMS:
        model = MODEL_PARAMS[model_name]["model"] (lr = lr,\
                                                    model_name = MODEL_PARAMS[model_name]["model_name"],
                                                    freeze = MODEL_PARAMS[model_name]["freeze"])
    else:
        raise Exception("Model Not Setup. Please configure your model in config.py")

    classifier = Classifier(DataManager = datamodule, model = model)

    classifier.train(lr = lr, epochs = epochs)

    classifier.test(on_train_set = False)


if __name__ == "__main__":

    ## python3 train.py -m pretrained_freeze -v 0 -e 1
    main()
    







