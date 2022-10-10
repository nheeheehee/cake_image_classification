import os
import torch
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from cake_classifier.dataset.make_dataset import DataManager

class Classifier:
    def __init__(self, DataManager, model, load_model = False):
        """
        Args:
        DataMananger: object from make_dataset.py which has attributes trainset and testset'
        """
        self.DataManager = DataManager
        self.trainloader = self.DataManager.trainloader
        self.testloader = self.DataManager.testloader
        self.classes = self.DataManager.classes 
        self.artifacts_dir = "./artifact/"

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.device = torch.device("cpu")
            # 'cuda:0' if torch.cuda.is_available() else 'cpu'
            # torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        # )

        # self.model = self.get_model() CHANGE TO GET MODEL
        # self.replace_model_last_layer(len(self.classes))
        self.model = model

        if load_model is True:
            self.load_model()
        
        self.model.to(self.device)
        self.create_loss_function()

    def create_loss_function(self):
        def custom_loss(y_pred_logits, y_true):
            """ Do what you want here, then return the loss """
            loss = None
            return loss
        
        # self.loss_function = custom_loss
        self.loss_function = nn.CrossEntropyLoss()
    
    def get_best_checkpoint(self):
        """
        Return:
            best_pt (str): Full path of .pt file in [artifact] folder with highest accuracy
        """
        best_acc = 0
        best_pt = None
        for filename in os.listdir("%s/checkpoint/" % self.artifacts_dir):
            self.load_model(filename)
            acc = self.test()
            if acc > best_acc:
                best_acc = acc
                best_pt = filename
                print("%s has current highest accuracy of %s" % (filename,acc))
        
        return best_pt

    
    def save_model(self):
        if not os.path.exists(self.artifacts_dir+"checkpoint/"):
            os.makedirs(self.artifacts_dir+"checkpoint/")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(
            self.model.state_dict(), 
            self.artifacts_dir+"checkpoint/model_%s.pt" % timestamp
        )
        print("model_%s.pt successfully saved!" % timestamp)
        return timestamp
    
    
    def load_model(self, filename=None):
        try:
            if filename is None:
                # get latest file, since files are named by date_time
                filename = sorted(
                    os.listdir(self.artifacts_dir+"checkpoint/")
                )[-1]
            self.model.load_state_dict(
                torch.load("%s/checkpoint/%s" % (self.artifacts_dir, filename))
            )
            print("%s successfully loaded..." % filename)
        except:
            print("Unable to load %s..." % filename)
    
    
    def set_optimizer(self):
        self.optimizer = \
            optim.Adam(self.model.parameters(), lr=self.lr)
    
    
    def set_scheduler(self):
        self.scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=1)
            
    def train(self, epochs=1, lr=1e-3, save=True):
        self.lr = lr
        self.set_optimizer()
        self.set_scheduler()
        
        history = []
        print("Beginning training for %d epochs" % epochs)
        print("lr: ", self.optimizer.param_groups[0]['lr'])
        loss_ema = 2.71828
        
        self.model.train()

        for epoch in range(epochs):
            for i, data in enumerate(self.trainloader):
                images, y_true = data
                images, y_true = images.to(self.device), y_true.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.loss_function(outputs, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                self.scheduler.step()
                
                loss_ema += 0.01*(loss.item() - loss_ema)  # corresponds to period of 100
                if (i+1) % 10 == 0:
                    history.append(loss_ema)
                    if (i+1) % 10 == 0:
                        # print every N mini-batches
                        print("Epoch %d Batch %d -- loss: %.3f" % (epoch+1, i+1, loss_ema))    
        
        if save is True:
            timestamp = self.save_model()
            if not os.path.exists(self.artifacts_dir+"history/"):
                os.makedirs(self.artifacts_dir+"history/")
            plt.plot(history)
            plt.title("EMA of loss")
            plt.savefig(self.artifacts_dir+"history/loss_%s.jpg"%timestamp)  # plt.show()
            
    
    def test(self, on_train_set=False, display=False):
        holder = {}
        holder['y_true'] = []
        holder['y_hat'] = []
        
        if on_train_set is True:
            print("Predicting on train set to get metrics")
            dataloader = self.trainloader
        else:
            print("Predicting on eval set to get metrics")
            dataloader = self.testloader
        
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                images, y_true = data
                images, y_true = images.to(self.device), y_true.to(self.device)
                
                outputs = self.model(images)
                _, y_hat = torch.max(outputs, 1)   # logits not required, index pos is sufficient
                holder['y_true'].extend(
                    list(y_true.cpu().detach().numpy())
                )
                holder['y_hat'].extend(
                    list(y_hat.cpu().detach().numpy())
                )
        
        y_true_all = holder['y_true']
        y_pred_all = holder['y_hat']
        M = confusion_matrix(y_true_all, y_pred_all)
        print("Confusion matrix: \n", M)
        print(classification_report(y_true_all, y_pred_all))
        



