import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config import CONFIG
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import check_accuracy, load_checkpoint, save_checkpoint

from classification.choice_loader import choice_loader_and_splits_dataset
from classification.metrics import get_metrics_collections
from classification.model.build_model import LitSystem, build_model


# pl.metrics.
def train_fn(loader,model,optimizer,loss_fn,metric_collection,scaler,device,epoch):
    
    logging.info (f"starting epoch {epoch}")
    loop=tqdm(loader)
    losses=[]
    # metrics={}
    for batch_idx, (data, targets) in enumerate(loop):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets) 
            losses.append(loss.item())
            

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
  
        with torch.no_grad():
            metric_value=metric_collection(scores.softmax(dim=-1),targets)
        print(metric_value)
        # metrics = {
        #     f"Accuracy": metric_value.item(),
        #     # f"{self.prefix}loss_dscrmn": loss_dscrmn.item(),
        #     # f"loss_pearson_coef": loss_dffclt_pearson_coef.item(),
        #     f"loss": loss.item()
        #     # f"{self.prefix}rsme": torch.sqrt(loss)
        # }
        
        loop.set_description(f"Epoch [{epoch}/{CONFIG.NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    loss_mean=np.mean(losses)
    wandb.log({"epoch":epoch,
               "loss": loss_mean,
               "precision":precision})

def main():
    
    
    wandb_logger = WandbLogger(project='TFM-classification',
                               entity='dcastf01',
                               offline=True, #to debug
                               
                               )
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=CONFIG.BATCH_SIZE,
                                                NUM_WORKERS=CONFIG.NUM_WORKERS)
    logging.info("DEVICE",CONFIG.DEVICE)
    train_loader=dataloaders["train"]
    test_loader=dataloaders["test"]
    loss_fn = nn.CrossEntropyLoss()
    metric_collection=get_metrics_collections(CONFIG.NUM_CLASSES, CONFIG.DEVICE)
    
    
    backbone=build_model(model_name=CONFIG.ModelName.torch_squeezenet)
    
    ##callbacks
    early_stopping=EarlyStopping(monitor='val_loss')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=CONFIG.PATH_CHECKPOINT,
        filename='SQUEEZENET-{epoch:02d}-{val_loss:.2f}',
        mode="min",
        save_last=True,
        save_top_k=3,
        
                        )
    
    
    
    model=LitSystem(backbone,metrics_collection=metric_collection,loss_fn=loss_fn)
    trainer=pl.Trainer(logger=wandb_logger,
                       gpus=-1,
                       max_epochs=CONFIG.NUM_EPOCHS,
                       precision=16,
                       limit_train_batches=0.005, #only to debug
                       limit_val_batches=0.005, #only to debug
                    #    val_check_interval=1,
                       log_gpu_memory=True,
                       callbacks=[
                            early_stopping ,
                            checkpoint_callback
                                  ]
                       )
    trainer.fit(model,train_loader,test_loader)
    
    
   
        
        

if __name__ == "__main__":

    main()
