import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import torchvision.transforms as transforms
from torchvision.models import resnet50, inception_v3, efficientnet_b2, mobilenet_v3_large, ResNet50_Weights, Inception_V3_Weights, EfficientNet_B2_Weights, MobileNet_V3_Large_Weights
from cake_classifier.models.interface import Model_Interface

class Pretrained(Model_Interface):
    """ Single Pretrained Model
    Set up replacement of last layers to all pretrained model to replace the final layer with 
    a number of fully connected layers with the correct num_classes"""

    def __init__(self, model_name, num_classes: int = 5, lr: float = 1e-4, dropout_rate: float = 0.3,
                freeze: Union[bool, int] = False):
        super(Pretrained, self).__init__(num_classes, lr, dropout_rate)
        
        self.device = torch.device(
            # 'cuda:0' if torch.cuda.is_available() else 'cpu'
            torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        )
        
        # self.model = model_name

        if model_name == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "inception":
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.model.aux_logits = False
        elif model_name == "efficient":
            self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        elif model_name == "mobile":
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        if isinstance(freeze, bool) and freeze == True:
            for params in self.model.parameters():
                params.requires_grad = False

        if isinstance(freeze, int):
            for params in self.model.parameters():
                params.requires_grad = False
                freeze -= 1
                if freeze == 0:
                    break

        last_layer_name, last_layer = list(self.model.named_modules())[-1]
        self.featveclen = last_layer.weight.shape[1]    


        if len(last_layer_name.split('.')) == 2:
            exec("self.model.%s[%s] = nn.Identity()" % \
                (last_layer_name.split('.')[0], last_layer_name.split('.')[1])
            )
        else:
            exec("self.model.%s = nn.Identity()" % \
                (last_layer_name,)
            )
        
        
        
        self.fc = nn.Linear(self.featveclen, self.num_classes)
        self.fc1 = nn.Linear(self.featveclen, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, inputs):
        if self.model == "inception":
            inputs = transforms.Resize(299)(inputs)
        x = self.model(inputs)
        # x = self.fc(x)

        return x

if __name__ == "__main__":
    model = Pretrained("resnet")
    x = torch.rand(32, 3, 224, 224)
    y = model(x)
    print(y.size())
        

        




        
        
