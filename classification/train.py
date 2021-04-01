import logging

import config
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_checkpoint, save_checkpoint
from classification.choice_loader import choice_loader_and_splits_dataset
from classification.model.torch_squeezeNet import get_squeezenet


def train_fn(loader,model,optimizer,loss_fn,scaler,device):
    
    logging.info ("starting ")
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets) #pendiente pasar los target a one hot
            

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    wandb.log({"epoch":epoch,
               "loss": loss})

def main():
    
    
    wandb.init(project='TFM-classification', entity='dcastf01')
    
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=config.BATCH_SIZE,
                                                NUM_WORKERS=config.NUM_WORKERS)
    logging.info("DEVICE",config.DEVICE)
    train_loader=dataloaders["train"]
    loss_fn = nn.CrossEntropyLoss()
    model=get_squeezenet(config.NUM_CLASSES).to(config.DEVICE)
    
    wandb.watch(model)
    optimizer= optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    

    
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader=train_loader,
                 model=model, 
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 scaler=scaler,
                 device=config.DEVICE)
        
        if config.SAVE_MODEL: 
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

            save_checkpoint(checkpoint,filename=config.CHECKPOINT_SQUEEZENET)
        
        
        
    raise NotImplementedError

if __name__ == "__main__":
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randint(5, (3,), dtype=torch.int64)
    main()
