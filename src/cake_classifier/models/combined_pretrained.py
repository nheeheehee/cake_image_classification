import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import torchvision.transforms as transforms
from torchvision.models import resnet50, inception_v3, efficientnet_b2, mobilenet_v3_large, ResNet50_Weights, Inception_V3_Weights, EfficientNet_B2_Weights, MobileNet_V3_Large_Weights
from cake_classifier.models.interface import Model_Interface
from cake_classifier.models.individual_pretrained import Pretrained

class Combined_Model(Model_Interface):
    """ Combined Pretrained Model
    Set up replacement of last layers to all pretrained model to replace the final layer with 
    a number of fully connected layers with the correct num_classes"""
    def __init__(self, model_name: list, num_classes: int = 5, lr: float = 1e-4, 
                dropout_rate: float = 0.3, freeze = False):
        super(Combined_Model, self).__init__(num_classes, lr, dropout_rate)

        if freeze == True:
            self.freeze_list = [True] * len(model_name)
        else:
            self.freeze_list = [False] * len(model_name)
        
        # self.device = torch.device(
        # # 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cpu")
        
        self.model1 = Pretrained(model_name[0], freeze = self.freeze_list[0])
        self.model2 = Pretrained(model_name[1], freeze = self.freeze_list[1])

        # self.model1.to(self.device)
        # self.model2.to(self.device)

        self.featveclen = self.model1.featveclen + self.model2.featveclen

        self.fc = nn.Linear(self.featveclen, num_classes)
        
        self.fc1 = nn.Linear(self.featveclen, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, inputs):
        x1 = self.model1(inputs)
        x2 = self.model2(inputs)
        x = torch.concat([x1,x2], dim = 1)

        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        out = self.fc3(x)
        # out = self.fc(x)

        return out

if __name__ == '__main__':
    model = Combined_Model(["resnet", "mobile"])
    x = torch.rand(32, 3, 224, 224)
    y = model(x)
    print(y.size())















    
