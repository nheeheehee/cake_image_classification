import torch.nn as nn 

class Model_Interface(nn.Module):
    """This is the Model Interface for Pytorch Models"""

    def __init__(self, 
                num_classes, 
                lr: float = 1e-4, 
                dropout_rate: float = 0.3):
        """_summary_ 
        
        Args: 
            num_classes (int):  Number of output classes
            lr (float, optional): learning rate for optimizer. Defaults to 1e-4.
            dropout_rate (float, optional): dropout rate. Defaults to 0.3

        """
        super(Model_Interface, self).__init__()

        self.num_classes = num_classes 
        self.learning_rate = lr 
        self.dropout = nn.Dropout (p = dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, inputs) -> NotImplementedError:
        '''Abstract forward function'''
        raise NotImplementedError


