from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cake_classifier.config import DATA_PATH
from cake_classifier.dataset.make_dataset import DataManager

def test_datamodule():
    '''
    To test function of datamodule
    Test items include:
    1. Sizes of train and test data are correct
    2. Batch size and shape of tensor are correct
    '''

    def get_sample_size(loader: DataLoader, batch_size: int):
        '''
        Get input and label batch size from 1 batch from data loader
        
        Args: 
            loader (Dataloader): DataLoader
            batch_size (int): DataLoader batch size
        
        Returns:
            Tuple: (torch.size(batch_inputs), torch.size(batch_labels)) 
        '''

        batch = next(iter(loader))
        images, labels = batch 
        return images.shape, labels.shape

    batch_size = 32
    test_module =  DataManager(batch_size = batch_size)   

    trainloader = test_module.trainloader
    testloader = test_module.testloader

    assert len(trainloader) == 118, "Trainset size mismatched" 
    assert len(testloader) == 40, "Testset size mismatched"

    assert get_sample_size(trainloader, batch_size) == (
        torch.Size([batch_size, 3, 224, 224]), torch.Size([batch_size])    
    ), "Trainloader Size Failed"

    assert get_sample_size(testloader, batch_size) == (
        torch.Size([batch_size, 3, 224, 224]), torch.Size([batch_size])
    ), "Testloader Size Failed"

if __name__ == "__main__":
    test_datamodule()





