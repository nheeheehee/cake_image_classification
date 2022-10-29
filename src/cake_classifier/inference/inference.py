'''
Module for API inference'''

from tkinter import Image
from typing import Union
from matplotlib.scale import LogitScale
import torch
import torch.nn.functional as F
from torchvision import transforms
from cake_classifier.config import MODEL_PARAMS, MAIN_PATH
from pathlib import Path
import io
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Predictor:
    ''' Class to do inference on the API'''

    def __init__(self, model_path: Union[str, Path]):
        '''
        Args:
        model_path (str): checkpoint to load the saved model
        
        '''

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if isinstance(model_path, Path):
            self.model_path = str(model_path)
        else:
            self.model_path = model_path
        self.model_type = "combined_freeze"
        self.model = MODEL_PARAMS[self.model_type]["model"](MODEL_PARAMS[self.model_type]["model_name"])

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval ()
        self.model.to(self.device)
    
    def transform(self, image_byte):

        transform_steps = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean, std)
        ])
        image = Image.open(io.BytesIO(image_byte))
        return transform_steps(image).unsqueeze(0)

    def predict(self, input):
        """
        Args:
            input (PIL Image): input a PIL image
        
        Returns:
            Dict: Prediction and Softmax probability of Prediction

        """

        # input = self.transform(input)
        input.to(self.device)


        with torch.no_grad():
            pred = self.model(input)

        logits = F.softmax(pred, dim = 1)
        probs = torch.max(logits, dim = 1)
        pred = torch.argmax(logits, dim = 1)

        return {"prediction": pred.item(), "probs": probs[0].item() }
    
    
if __name__ == "__main__":
    model_path = MAIN_PATH /"artifact"/"checkpoint"/"model_20221010_200315.pt"
    img_path = MAIN_PATH / "choc.jpg"
    predictor = Predictor(model_path)



        
