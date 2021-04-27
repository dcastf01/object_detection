
import pandas as pd
import torch
from config import CONFIG

from torch.utils.data import DataLoader
from classification.augmentation import get_transform_from_aladdinpersson
from classification.loaders.compcars_loader import CompcarLoader,CompcarLoaderTripletLoss
from classification.loaders.cars196_loader import Cars196Loader,Cars196LoaderTripletLoss
import logging
from enum import Enum

class Dataset (Enum):
    compcars=1
    cars196=2

def choice_loader_and_splits_dataset(name_dataset:Enum,
                                     BATCH_SIZE:int=16,
                                     NUM_WORKERS:int=0,
                                     use_tripletLoss:bool=False) -> dict:
    
    transforms=get_transform_from_aladdinpersson()
    train_transform=transforms["train"]
    val_transform=transforms["val"]
    test_transform=transforms["test"]
        
    if name_dataset==Dataset.cars196:
        all_ds=pd.read_csv(CONFIG.DATASET.CARS196.PATH_CSV)
        train_ds=all_ds[all_ds["split"]=="train"]
        test_ds=all_ds[all_ds["split"]=="test"]
        
        if use_tripletLoss:
            logging.info("loading cars196 triplet loss")
            loader=Cars196LoaderTripletLoss
        else:
            loader=Cars196Loader
        
        train_dataset=loader(train_ds,
                        root_dir_images=CONFIG.DATASET.CARS196.PATH_IMAGES,
                        transform=train_transform,
                        )
        
        test_dataset=loader(test_ds,
                        root_dir_images=CONFIG.DATASET.CARS196.PATH_IMAGES,
                        transform=test_transform,                        
                        )
        
        
    elif name_dataset ==Dataset.compcars:
        def expand_information_useful_on_txt(df:pd.DataFrame)->pd.DataFrame:
            df[["make_id","model_id","released_year","filename"]]=df["Filepath"].str.split("/",expand=True)
            return df
              
        train_ds = pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TRAIN_REVISITED)
        train_ds=expand_information_useful_on_txt(train_ds)

        test_ds=pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TEST_REVISITED,)
        test_ds=expand_information_useful_on_txt(test_ds)

        if use_tripletLoss:
            logging.info("loading compcar triplet loss")
            loader=CompcarLoaderTripletLoss
            BATCH_SIZE=BATCH_SIZE//2
        else:
            loader=CompcarLoader
        train_dataset=loader(train_ds,
                        root_dir_images=CONFIG.DATASET.COMPCAR.PATH_IMAGES,
                        transform=train_transform,
                        )
        
        test_dataset=loader(test_ds,
                        root_dir_images=CONFIG.DATASET.COMPCAR.PATH_IMAGES,
                        transform=test_transform,                        
                        )
   
    else:
        raise NotImplementedError
      
    
    train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=CONFIG.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=True,
                    # sampler=
                                                    )

    test_dataset_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=CONFIG.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=True,
                )
    dataloaders = {
        "train": train_dataset_loader,
        "test": test_dataset_loader,
    }
    return dataloaders


def test_choice_loader():
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                BATCH_SIZE=CONFIG.BATCH_SIZE,
                                                # NUM_WORKERS=CONFIG.NUM_WORKERS
                                                )
    
# test_choice_loader()