import torch
import torch.nn as nn
from config import CONFIG
from typing import Union
from classification.model.torch_squeezeNet import get_squeezenet


import pytorch_lightning as pl

def build_model(model_name:str,loss_fn=None,metrics:Union[None,list]=None):
    if model_name==CONFIG.ModelName.torch_squeezenet:
        backbone=get_squeezenet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
        
    model=backbone
        
        
    return model
    
    
class LitSystem(pl.LightningModule):
    def __init__(self,model,loss_fn=nn.CrossEntropyLoss(),metrics=):
        super().__init__()
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.model=model
        self.loss_fn=loss_fn
        
    def forward(self,x):
        
        x=self.model(x)
        
        return x
    
    def training_step(self,batch,batch_idx):
        x,targets=batch
        
        preds=self.model(x)
        
        loss=self.loss_fn(preds,targets)
        
        self.log('train_loss',loss)
        
        return loss
    def configure_optimizers(self):
        
        optimizer= torch.optim.Adam(self.parameters(), lr=CONFIG.LEARNING_RATE)
        return optimizer

    