
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from config import CONFIG
import logging
from classification.metrics import get_metrics_collections_base,get_metric_AUROC


class LitSystem(pl.LightningModule):
    def __init__(self,
                 model,
                 
                #   loss_fn=nn.CrossEntropyLoss(),
                  lr=CONFIG.LEARNING_RATE):
        
        super().__init__()
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.model=model
        # self.loss_fn=loss_fn
        metrics_base=get_metrics_collections_base(NUM_CLASS=CONFIG.NUM_CLASSES)
        self.train_metrics_base=metrics_base.clone(prefix="train")
        # self.train_metric_auroc=get_metric_AUROC(NUM_CLASS=CONFIG.NUM_CLASSES)
        self.valid_metrics_base=metrics_base.clone(prefix="valid")
        # self.valid_metric_auroc=get_metric_AUROC(NUM_CLASS=CONFIG.NUM_CLASSES)
        
        # log hyperparameters
        self.save_hyperparameters()
        
        self.lr=lr
            
    def forward(self,x):
        
        x=self.model(x)
        
        return x #quizá existe el problema de que la salida es un diccionario
    
    def on_epoch_start(self):
        torch.cuda.empty_cache()
    
    def training_step(self,batch,batch_idx):
        x,targets=batch
        
        data_dict=self.model(x,targets)
        loss=data_dict["loss"]
        preds=data_dict["preds"]
        data_dict.pop("preds")

        if torch.any(torch.isnan(preds)):
            nan_mask=torch.any(torch.isnan(preds))
            logging.error((preds))
            print("tiene nan, averiguar")
            print( "resultado de softmax", nn.functional.softmax(preds,dim=1))
            logging.error(nn.functional.softmax(preds,dim=1))
            raise RuntimeError(f"Found NAN in output {batch_idx} at indices: ", nan_mask.nonzero(), "where:", x[nan_mask.nonzero()[:, 0].unique(sorted=True)])

        if isinstance(targets,list):
            targets=targets[0]
        #lo ideal sería hacer algo tipo loss,preds=self.model(x) de esta manera al modelo se le 
        #incluiria el loss fn y así podremos hacer la tripleta
        # loss=self.loss_fn(preds,targets)
        preds_probability=nn.functional.softmax(preds,dim=1)
        metric_value=self.train_metrics_base(preds_probability,targets)
        data_dict={**data_dict,**metric_value}
        # metric_value={**metric_value,
        #               **self.train_metric_auroc(preds.softmax(dim=1),targets)}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        # self.log('train_loss',loss)
        # self.log('train_metrics',metric_value)
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, targets = batch
        
        data_dict=self.model(x,targets)
        preds=data_dict["preds"]
        data_dict.pop("preds")
        data_dict=self.add_prefix_into_dict_only_loss(data_dict,"val")
        # loss=data_dict["loss"]
        # data_dict.pop("loss")
        # data_dict["val_loss"]=loss
        
        
        if isinstance(targets,list):
            targets=targets[0]
        preds_probability=nn.functional.softmax(preds,dim=1)
        a=torch.sum(preds_probability,dim=1)
        metric_value=self.valid_metrics_base(preds_probability,targets)
        # metric_value={**metric_value,
                    #   **self.valid_metric_auroc(preds.softmax(dim=0),targets)}
        
        data_dict={**data_dict,**metric_value}
        #########CREAR UNA FUNCIÓN QUE COJA METRICA Y DATA DICT, Y SUELTE LA PREDICCION 
        # ##################Y GENERE EL LOG CORRESPONDIENTE
        # Log validation loss (will be automatically averaged over an epoch)
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        # self.log('val_loss', loss, on_step=False, on_epoch=True)
        # self.log('val_metrics',metric_value, on_step=False, on_epoch=True)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))
   
            
    def configure_optimizers(self):
        
        optimizer= torch.optim.SGD(self.parameters(), lr=self.lr)
            

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG.NUM_EPOCHS, eta_min=5)
        return [optimizer], [scheduler]

    
    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix:str):
 
        on_step=False
        on_epoch=True 
        
        for metric,value in data_dict.items():
            if metric != "preds":
                if "loss" in metric.split("_"):
                    self.log("_".join([prefix,metric]),value,
                            on_step=on_step, 
                            on_epoch=on_epoch, 
                            sync_dist=True,
                            logger=True)
                else:
                    self.log("_".join([prefix,metric]),value,
                            on_step=on_step, 
                            on_epoch=on_epoch, 
                            logger=True
                    )
    def add_prefix_into_dict_only_loss(self,data_dict:dict,prefix:str=""):
        data_dict_aux={}
        for k,v in data_dict.items():            
            data_dict_aux["_".join([prefix,k])]=v
            
        return data_dict_aux