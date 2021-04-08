
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from config import CONFIG
from dataset.compcars.compcar_analisis import CompcarAnalisis
from PIL import Image
# from skimage.viewer import ImageViewer
from torch.utils.data import DataLoader
from classification.augmentation import get_transform_from_aladdinpersson
from classification.loaders.compcars_loader import CompcarLoader


def choice_loader_and_splits_dataset(name_dataset:str,BATCH_SIZE:int=16,NUM_WORKERS:int=1) -> dict:
    
    if name_dataset.lower()=="cars196":
        raise NotImplementedError
    elif name_dataset.lower() =="compcars":
        def expand_information_useful_on_txt(df:pd.DataFrame)->pd.DataFrame:
            df[["make_id","model_id","released_year","filename"]]=df["Filepath"].str.split("/",expand=True)
            return df
              
        train_ds = pd.read_csv(CONFIG.PATH_COMPCAR_TRAIN_REVISITED)
        train_ds=expand_information_useful_on_txt(train_ds)

        test_ds=pd.read_csv(CONFIG.PATH_COMPCAR_TEST_REVISITED,)
        test_ds=expand_information_useful_on_txt(test_ds)

        transforms=get_transform_from_aladdinpersson()
        train_transform=transforms["train"]
        val_transform=transforms["val"]
        test_transform=transforms["test"]
        
        train_dataset=CompcarLoader(train_ds,
                         root_dir_images=CONFIG.PATH_COMPCAR_IMAGES,
                         transform=train_transform,
                       
                         )
        
        test_dataset=CompcarLoader(test_ds,
                         root_dir_images=CONFIG.PATH_COMPCAR_IMAGES,
                         transform=test_transform,
                         )
   
    else:
        raise NotImplementedError
 
    train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=True
                    # sampler=
                                                    )

    test_dataset_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=True
                )
    dataloaders = {
        "train": train_dataset_loader,
        "test": test_dataset_loader,
    }
    return dataloaders


def test_choice_loader():
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=CONFIG.BATCH_SIZE,
                                                NUM_WORKERS=CONFIG.NUM_WORKERS)
    
# test_choice_loader()