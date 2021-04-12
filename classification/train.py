import datetime
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

from classification.callback import ConfusionMatrix_Wandb
from classification.choice_loader import choice_loader_and_splits_dataset
from classification.metrics import get_metrics_collections
from classification.model.build_model import build_model

from classification.lit_system import LitSystem


def main():
    
    
    wandb_logger = WandbLogger(project='TFM-classification',
                               entity='dcastf01',
                               name=str(datetime.datetime.now()),
                            #    offline=True, #to debug
                               )
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=CONFIG.BATCH_SIZE,
                                                NUM_WORKERS=CONFIG.NUM_WORKERS,
                                                use_tripletLoss=True
                                                )
    logging.info("DEVICE",CONFIG.DEVICE)
    train_loader=dataloaders["train"]
    test_loader=dataloaders["test"]
    
    
    loss_fn = nn.CrossEntropyLoss()
    metric_collection=get_metrics_collections(CONFIG.NUM_CLASSES, CONFIG.DEVICE)
    
    
    
    
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
    
    # confusion_matrix_wandb=ConfusionMatrix_Wandb(list(range(CONFIG.NUM_CLASSES)))
        
    backbone=build_model(model_name=CONFIG.ARCHITECTURES_AVAILABLE.torch_transFG,
                         loss_fn=loss_fn)
    model=LitSystem(backbone,
                    metrics_collection=metric_collection,
                    # loss_fn=loss_fn,
                    lr=CONFIG.LEARNING_RATE,
                    )
    trainer=pl.Trainer(logger=wandb_logger,
                       gpus=-1,
                       max_epochs=CONFIG.NUM_EPOCHS,
                       precision=16,
                       limit_train_batches=0.00005, #only to debug
                       limit_val_batches=0.005, #only to debug
                    #    val_check_interval=1,
                       log_gpu_memory=True,
                       callbacks=[
                            # early_stopping ,
                            # checkpoint_callback,
                            # confusion_matrix_wandb
                                  ]
                       )
    trainer.fit(model,train_loader,test_loader)
         

if __name__ == "__main__":

    main()
