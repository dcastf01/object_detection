import logging

from config import CONFIG
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_checkpoint, save_checkpoint,check_accuracy
from classification.choice_loader import choice_loader_and_splits_dataset
from classification.model.torch_squeezeNet import get_squeezenet


def train_fn(loader,model,optimizer,loss_fn,scaler,device,epoch):
    
    logging.info (f"starting epoch {epoch}")
    loop=tqdm(loader)
    losses=[]
    accuracies=[]
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
        
        loop.set_description(f"Epoch [{epoch}/{CONFIG.NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    
    wandb.log({"epoch":epoch,
               "loss": loss})

def main():
    
    
    wandb.init(project='TFM-classification', entity='dcastf01')
    
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=CONFIG.BATCH_SIZE,
                                                NUM_WORKERS=CONFIG.NUM_WORKERS)
    logging.info("DEVICE",CONFIG.DEVICE)
    train_loader=dataloaders["train"]
    test_loader=dataloaders["test"]
    loss_fn = nn.CrossEntropyLoss()
    model=get_squeezenet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
    
    wandb.watch(model)
    optimizer= optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    if CONFIG.LOAD_MODEL:
        load_checkpoint(torch.load(f=CONFIG.CHECKPOINT_SQUEEZENET), model, optimizer)
        # check_accuracy(test_loader, model, CONFIG.DEVICE)
    
    
    for epoch in range(CONFIG.NUM_EPOCHS):
        train_fn(loader=train_loader,
                 model=model, 
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 scaler=scaler,
                 device=CONFIG.DEVICE,
                 epoch=epoch)
        
        # check_accuracy(test_loader, model, CONFIG.DEVICE)
        
        if CONFIG.SAVE_MODEL: 
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            
            
            save_checkpoint(checkpoint,filename=CONFIG.CHECKPOINT_SQUEEZENET)
        
        
        
        
        

if __name__ == "__main__":
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randint(5, (3,), dtype=torch.int64)
    main()
