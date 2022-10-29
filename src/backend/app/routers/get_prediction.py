'''Router definition for API'''

import pathlib

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image

from backend.app.config import Settings, get_settings
from cake_classifier.inference.inference import Predictor

model_path = pathlib.Path(__file__).resolve().parents[0]/ "deploy"/ "model_20221024_234744.pt"

predictor = Predictor(str(model_path))

app = APIRouter(prefix = '/cake_classifier')

@app.get("/")

def hello():
    '''Base Input'''
    return {'Model Name': "resnet 50 + mobilenet"}

@app.get("/env")

def get_env(settings: Settings = Depends(get_settings)):
    '''check env'''
    return {'Env': Settings.environment}

@app.post("/predict")

def upload (file: UploadFile = File(...)):
    '''Get prediction
    
    Args:
        file (UploadFile, optional): Upload image file. to convert pil image and use model to predict class
    
    Returns:
        prediction (Dict): {Predicted class, predicted probability}
    
    '''
    # with open(file.file, 'rb') as f:
    #     sample = f.read()

    # output = predictor.transform(sample)    
    # output = predictor.predict(output)
    # return output
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        
        with open(file.filename, 'rb') as f:
            sample = f.read()
            output = predictor.transform(sample)    
            output = predictor.predict(output)
            return output

    except Exception:
        return {"message": "There was an error uploading"}
    
    finally: 
        file.file.close()
