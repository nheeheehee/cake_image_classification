import pathlib
from PIL import Image
from cake_classifier.inference.inference import Predictor 
from cake_classifier.config import MAIN_PATH

def test_inference():
    """Input a sample image to generate prediction and probability"""

    model_path = MAIN_PATH /"artifact"/"checkpoint"/"model_20221024_234744.pt"
    img_path = MAIN_PATH / "choc.jpg"
    predictor = Predictor(model_path)
    
    with open(img_path, 'rb') as f:
        image_byte = f.read()
    
    input = predictor.transform(image_byte)
    output = predictor.predict(input)

    print(output)
    assert output["prediction"] in ['apple_pie', 'cheesecake', 'chocolate_cake', 'french_toast', 'garlic_bread'], "Invalid Prediction"
    assert (output["probs"] <= 1) and (output["probs"] >= 0), "Invalid Probability"


if __name__ == "__main__":
    test_inference()

