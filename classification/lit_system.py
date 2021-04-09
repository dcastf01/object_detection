
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from config import CONFIG


class LitSystem(pl.LightningModule):
    def __init__(self,
                 model,
                 metrics_collection:torchmetrics.MetricCollection,
                #   loss_fn=nn.CrossEntropyLoss(),
                  lr=CONFIG.LEARNING_RATE):
        
        super().__init__()
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.model=model
        # self.loss_fn=loss_fn
        self.train_metrics=metrics_collection.clone(prefix="train_metrics")
        self.valid_metrics=metrics_collection.clone(prefix="valid_metrics")
        
        # log hyperparameters
        self.save_hyperparameters()
        
        self.lr=lr
            
    def forward(self,x):
        
        x=self.model(x)
        
        return x
    
    def training_step(self,batch,batch_idx):
        x,targets=batch
        
        loss,preds=self.model(x,targets)
        
        #lo ideal sería hacer algo tipo loss,preds=self.model(x) de esta manera al modelo se le 
        #incluiria el loss fn y así podremos hacer la tripleta
        # loss=self.loss_fn(preds,targets)
        
        metric_value=self.train_metrics(preds.softmax(dim=-1),targets)
        
        self.log('train_loss',loss)
        self.log('train_metrics',metric_value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, targets = batch
        
        loss,preds=self.model(x,targets)
        
        # loss = self.loss_fn(preds, targets)

        metric_value=self.valid_metrics(preds.softmax(dim=-1),targets)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_metrics',metric_value, on_step=False, on_epoch=True)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))
   
            
    def configure_optimizers(self):
        
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    