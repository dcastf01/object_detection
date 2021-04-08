import torch
import torch.nn as nn
from config import CONFIG
from typing import Union
from classification.model.torch_squeezeNet import get_squeezenet

import torchmetrics
import pytorch_lightning as pl

def build_model(model_name:str,loss_fn=None,metrics:Union[None,list]=None):
    if model_name==CONFIG.ModelName.torch_squeezenet:
        backbone=get_squeezenet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
        
    model=backbone
        
        
    return model
    
    
class LitSystem(pl.LightningModule):
    def __init__(self,
                 model,
                 metrics_collection:torchmetrics.MetricCollection,
                  loss_fn=nn.CrossEntropyLoss(),):
        
        super().__init__()
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.model=model
        self.loss_fn=loss_fn
        self.train_metrics=metrics_collection.clone(prefix="train_metrics")
        self.valid_metrics=metrics_collection.clone(prefix="valid_metrics")
            
    def forward(self,x):
        
        x=self.model(x)
        
        return x
    
    def training_step(self,batch,batch_idx):
        x,targets=batch
        
        preds=self.model(x)
        
        loss=self.loss_fn(preds,targets)
        
        metric_value=self.train_metrics(preds.softmax(dim=-1),targets)
        
        self.log('train_loss',loss)
        self.log('train_metrics',metric_value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, targets = batch
        
        preds = self.model(x)
        
        loss = self.loss_fn(preds, targets)

        metric_value=self.valid_metrics(preds.softmax(dim=-1),targets)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('val_loss', loss)
        self.log('val_metrics',metric_value)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))
        
        
    def configure_optimizers(self):
        
        optimizer= torch.optim.Adam(self.parameters(), lr=CONFIG.LEARNING_RATE)
        return optimizer

    